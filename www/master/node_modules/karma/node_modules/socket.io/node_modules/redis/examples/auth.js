var redis  = require("redis"),
    client = redis.createClient();

// This command is magical.  Client stashes the password and will issue on every connect.
client.auth("somepass");
