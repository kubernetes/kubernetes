require("http").createServer(function (req, res) {
  res.writeHead(200, {"content-type":"application/json"})
  res.end(JSON.stringify({ok: true}))
}).listen(1337)
