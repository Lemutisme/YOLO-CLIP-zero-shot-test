#建立一个检索字典，similarities相似值去获取标题和链接
with torch.no_grad():
      # image_features = model.encode_image(image)
      text_features = model.encode_text(text)
      text_features /= text_features.norm(dim=-1, keepdim=True)
      # text_features = text_features.cpu().numpy()
      # print(text_features)
      # print(image_features.shape)
      # print(image_features)
      # print(text_features.shape)
      similarities = (image @ text_features.T).squeeze(1)
      print(similarities[:, 0])
      best_photo_idx = np.argsort(similarities[:, 0].numpy())[::-1]

      print(best_photo_idx)
      rank_results = [aidss[i] for i in best_photo_idx[:10]]
      titles1 = []
      pics1 = []
      for j in rank_results:
          titles1.append(kkk_dict_all[j][0])
          pics1.append(kkk_dict_all[j][1])
      print(titles1, pics1)

  return render_template('display1.html', query=query, lis1=titles[:10], lis2=pics[:10], lis3=titles1, lis4=pics1)
