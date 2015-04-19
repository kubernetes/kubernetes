var redis = require("redis"),
    client = redis.createClient();

client.sadd("mylist", 1);
client.sadd("mylist", 2);
client.sadd("mylist", 3);

client.set("weight_1", 5);
client.set("weight_2", 500);
client.set("weight_3", 1);

client.set("object_1", "foo");
client.set("object_2", "bar");
client.set("object_3", "qux");

client.sort("mylist", "by", "weight_*", "get", "object_*", redis.print);
// Prints Reply: qux,foo,bar