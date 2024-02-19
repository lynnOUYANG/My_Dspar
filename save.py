data = data.to('cuda')
    infer_time_list = []
    test_acc_list = []
    train_time_list = []
    for run in range(args.runs):
        data.edge_index = edge_index
        data.adj_t = adj_t
        # print(data)
        # exit()
        model.reset_parameters()
        optimizer = get_optimizer(model_config, model)
        best_val_acc = 0.0
        patience = 0
        my_ebd = None
        state = 'pre'
        if args.amp:
            print('activate amp mode')
            scaler = GradScaler()
        else:
            scaler = None
        for epoch in range(1, 1 + model_config['epochs']):
            # edge_drop = model_config.get('edge_drop', 0.0)
            # if edge_drop > 0:
            #     adj_t = drop_edge(data.adj_t, edge_drop).to('cuda')
            # ==== train the model ====
            # if model_config.get('adjust_lr', False):
            #     adjust_learning_rate(optimizer, model_config['lr'], epoch)
            if model_config['name'] == 'EbdGNN':
                # torch.cuda.reset_max_memory_allocated()
                # 运行你的代码
                a = time.time()

                loss = ebd_train(model, optimizer, data, grad_norm, scaler, args, state=state,
                                 ebd=my_ebd)
                train_time = time.time() - a
                if (model_config['name'] == 'EbdGNN' and epoch > model_config['pepochs']) or model_config[
                    'name'] != 'EbdGNN':
                    train_time_list.append(train_time)

            else:
                loss = train(model, optimizer, data, grad_norm, scaler, args)
            # ===========================
            if (epoch % 100) == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Train Loss: {loss:.4f}')
            if model_config['name'] == 'EbdGNN':

                # 运行你的代码

                result = ebd_test(model, data, evaluator, state, ebd=my_ebd)

                max_memory_used = torch.cuda.max_memory_allocated()
                # print(f"Max memory used by the test : {max_memory_used / 1024 / 1024}MB")
                train_acc, valid_acc, test_acc, pred, ebd, infer_time = result

            else:
                result = test(model, data, evaluator,args.amp)
                train_acc, valid_acc, test_acc, infer_time = result
            if (model_config['name'] == 'EbdGNN' and epoch > model_config['pepochs']) or model_config[
                'name'] != 'EbdGNN':
                infer_time_list.append(infer_time)

            result = (train_acc, valid_acc, test_acc)
            if valid_acc > best_val_acc:
                patience = 0
                best_val_acc = valid_acc
                best_epoch = epoch
                if model_config['name'] == 'EbdGNN':
                    best_ebd = ebd
                    result_s = ebd_test(model, data, evaluator, state, ebd=my_ebd)
                    _, _, test_acc, best_pred, ebd, infer_time = result_s
                else:
                    result_t = test(model, data, evaluator)
                    _, _, test_acc, infer_time = result_t
            else:
                patience += 1
                if patience > 100:
                    if model_config['name'] == 'EbdGNN':
                        if epoch > model_config['pepochs'] + 128:
                            break
                    else:
                        break

            test_acc_list.append(test_acc)
            logger.add_result(run, result)
            if (model_config['name'] == 'EbdGNN' and epoch > model_config['pepochs']) or model_config[
                'name'] != 'EbdGNN':
                infer_time_list.append(infer_time)
            if epoch%100==0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Train f1: {100 * train_acc:.2f}%, '
                      f'Valid f1: {100 * valid_acc:.2f}% '
                      f'Test f1: {100 * test_acc:.2f}%')

        if model_config['name'] == 'EbdGNN' and epoch == model_config['pepochs']:
            state = 'train'
            print("start sparse")
            similarity = model.node_similarity(best_pred, data)
            data = model.graph_sparse(similarity, data, args, device=model_config['device'])
            my_ebd = best_ebd

        logger.print_statistics(run)
    logger.print_statistics()
    infer_time_list = np.array(infer_time_list)
    infer_time_mean = infer_time_list.mean()
    print(f"the mean infer time of each epoch is :{infer_time_mean:.5f}")
    train_time_list = np.array(train_time_list)
    train_time_mean = train_time_list.mean()
    print(f"the mean train time of each epoch is :{train_time_mean:.5f}")

    test_acc_list = np.array(test_acc_list)
    test_acc_mean = test_acc_list.max()

    if torch.cuda.is_available():
        print("Max GPU memory usage: {:.5f} GB, max GPU memory cache {:.5f} GB".format(
            torch.cuda.max_memory_allocated(args.gpu) / (1024 ** 3), torch.cuda.max_memory_reserved() / (1024 ** 3)))
    my_model_name = f"products_{args.gnn_model}.pth"
    torch.save(model.state_dict(), my_model_name)