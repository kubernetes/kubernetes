var client = require("redis").createClient();

client.mget(["sessions started", "sessions started", "foo"], function (err, res) {
    console.dir(res);
});