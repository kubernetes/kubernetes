var client = require("redis").createClient();

function print_results(obj) {
    console.dir(obj);
}

// build a map of all keys and their types
client.keys("*", function (err, all_keys) {
    var key_types = {};
    
    all_keys.forEach(function (key, pos) { // use second arg of forEach to get pos
        client.type(key, function (err, type) {
            key_types[key] = type;
            if (pos === all_keys.length - 1) { // callbacks all run in order
                print_results(key_types);
            }
        });
    });
});
