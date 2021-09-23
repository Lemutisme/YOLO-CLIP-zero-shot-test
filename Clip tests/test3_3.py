##flask &&有道翻译
data = {
      "i": query,
      "from": "AUTO",
      "to": "AUTO",
      "smartresult": "dict",
      "client": "fanyideskweb",
      "doctype": "json",
      "version": "2.1",
      "keyfrom": "fanyi.web",
      "action": "FY_BY_REALTIME",
      "typoResult": "false"
  }
  response = requests.post("http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule", data=data).json()
  # print(response)
  # print(type(response))
  bb = response["translateResult"][0][0]["tgt"]
  print(bb)
