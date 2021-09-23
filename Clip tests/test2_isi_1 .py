@app.route('/findmore/<username>', methods=['GET'])
def myinfohtml(username):
    print(username)
    # image to image
    source_image = r"D:***图\{}.jpg".format(username)
    with torch.no_grad():
        image_feature = model.encode_image(preprocess(Image.open(source_image)).unsqueeze(0).to(device))
        image_feature = (image_feature / image_feature.norm(dim=-1, keepdim=True)).cpu().numpy()
    best_photo_ids1 = (image @ image_feature.T).squeeze(1)
    best_photo_idx2 = np.argsort(best_photo_ids1[:, 0].numpy())[::-1]
    rank_results2 = [aidss[i] for i in best_photo_idx2[:30]]
    # 召回结果封装
    titles3 = []
    pics3 = []
    for j in rank_results2:
        titles3.append(kkk_dict_all[j][0])
        pics3.append(kkk_dict_all[j][1])
    print(titles3, pics3)

    return render_template('display3.html', lis31=titles3, lis32=pics3)
