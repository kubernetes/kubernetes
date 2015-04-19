var redis = require("redis"),
    client = redis.createClient();

// Extend the RedisClient prototype to add a custom method
// This one converts the results from "INFO" into a JavaScript Object

redis.RedisClient.prototype.parse_info = function (callback) {
    this.info(function (err, res) {
        var lines = res.toString().split("\r\n").sort();
        var obj = {};
        lines.forEach(function (line) {
            var parts = line.split(':');
            if (parts[1]) {
                obj[parts[0]] = parts[1];
            }
        });
        callback(obj)
    });
};

client.parse_info(function (info) {
    console.dir(info);
    client.quit();
});
