var client = require("redis").createClient(),
    client2 = require("redis").createClient();

client.subscribe("something");
client.on("subscribe", function (channel, count) {
    console.log("Got sub: " + channel);
    client.unsubscribe("something");
});

client.on("unsubscribe", function (channel, count) {
    console.log("Got unsub: " + channel + ", quitting");
    client.quit();
});

// exercise unsub before sub
client2.unsubscribe("something");
client2.subscribe("another thing");
client2.quit();
