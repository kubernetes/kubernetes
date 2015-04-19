var fs = require('graceful-fs');
var intersect = require('intersect');

// Function to check if a file is a component(1) file
function isComponent(file, callback) {
    fs.readFile(file, function (err, contents) {
        var json;
        var keys;
        var common;

        // If an error occurs while reading the file, we ignore it
        if (err) {
            return callback(false);
        }

        try {
            json = JSON.parse(contents.toString());
        } catch (err) {
            return callback(false);
        }

        // Attempt to find specific things from the component(1) spec
        // Note that we don't parse the dependencies property because at any point
        // we can allow / to specify directories
        // Bellow only some clearly not ambiguous properties are checked
        keys = Object.keys(json);
        common = intersect(keys, [
            'repo',
            'development',
            'local',
            'remotes',
            'paths',
            'demo'
        ]);

        // If none were found, than it's a valid component.json bower file
        callback(common.length ? true : false);
    });
}

module.exports = isComponent;
