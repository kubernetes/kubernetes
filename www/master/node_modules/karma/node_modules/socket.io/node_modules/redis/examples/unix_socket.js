var redis = require("redis"),
    client = redis.createClient("/tmp/redis.sock"),
    profiler = require("v8-profiler");

client.on("connect", function () {
    console.log("Got Unix socket connection.")
});

client.on("error", function (err) {
    console.log(err.message);
});

client.set("space chars", "space value");

setInterval(function () {
    client.get("space chars");
}, 100);

function done() {
    client.info(function (err, reply) {
        console.log(reply.toString());
        client.quit();
    });
}

setTimeout(function () {
    console.log("Taking snapshot.");
    var snap = profiler.takeSnapshot();
}, 5000);
