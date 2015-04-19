var client = require("redis").createClient();

client.set("foo", "barvalskdjlksdjflkdsjflksdjdflkdsjflksdjflksdj", function (err, res) {
    if (err) {
        console.log("Got an error, please adapt somehow.");
    } else {
        console.log("Got a result: " + res);
    }
});

client.quit();
