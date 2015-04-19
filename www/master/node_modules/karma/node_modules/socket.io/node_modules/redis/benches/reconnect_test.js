var redis = require("../index").createClient(null, null, {
//    max_attempts: 4
});

redis.on("error", function (err) {
    console.log("Redis says: " + err);
});

redis.on("ready", function () {
    console.log("Redis ready.");
});

redis.on("reconnecting", function (arg) {
    console.log("Redis reconnecting: " + JSON.stringify(arg));
});
redis.on("connect", function () {
    console.log("Redis connected.");
});

setInterval(function () {
    var now = Date.now();
    redis.set("now", now, function (err, res) {
        if (err) {
            console.log(now + " Redis reply error: " + err);
        } else {
            console.log(now + " Redis reply: " + res);
        }
    });
}, 100);
