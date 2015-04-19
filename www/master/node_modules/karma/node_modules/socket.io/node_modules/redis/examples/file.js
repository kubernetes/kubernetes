// Read a file from disk, store it in Redis, then read it back from Redis.

var redis = require("redis"),
    client = redis.createClient(),
    fs = require("fs"),
    filename = "kids_in_cart.jpg";

// Get the file I use for testing like this:
//    curl http://ranney.com/kids_in_cart.jpg -o kids_in_cart.jpg
// or just use your own file.

// Read a file from fs, store it in Redis, get it back from Redis, write it back to fs.
fs.readFile(filename, function (err, data) {
    if (err) throw err
    console.log("Read " + data.length + " bytes from filesystem.");
    
    client.set(filename, data, redis.print); // set entire file
    client.get(filename, function (err, reply) { // get entire file
        if (err) {
            console.log("Get error: " + err);
        } else {
            fs.writeFile("duplicate_" + filename, reply, function (err) {
                if (err) {
                    console.log("Error on write: " + err)
                } else {
                    console.log("File written.");
                }
                client.end();
            });
        }
    });
});
