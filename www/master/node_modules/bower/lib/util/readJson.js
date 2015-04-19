var path = require('path');
var bowerJson = require('bower-json');
var Q = require('q');

// The valid options are the same as bower-json#readFile.
// If the "assume" option is passed, it will be used if no json file was found

// This promise is resolved with [json, deprecated, assumed]
// - json: The read json
// - deprecated: The deprecated filename being used or false otherwise
// - assumed: True if a dummy json was returned if no json file was found, false otherwise
function readJson(file, options) {
    options = options || {};

    // Read
    return Q.nfcall(bowerJson.read, file, options)
    .spread(function (json, jsonFile) {
        var deprecated;

        jsonFile = path.basename(jsonFile);
        deprecated = jsonFile === 'component.json' ? jsonFile : false;

        return [json, deprecated, false];
    }, function (err) {
        // No json file was found, assume one
        if (err.code === 'ENOENT' && options.assume) {
            return [bowerJson.parse(options.assume, options), false, true];
        }

        err.details = err.message;

        if (err.file) {
            err.message = 'Failed to read ' + err.file;
            err.data = { filename: err.file };
        } else {
            err.message = 'Failed to read json from ' + file;
        }

        throw err;
    });
}

module.exports = readJson;
