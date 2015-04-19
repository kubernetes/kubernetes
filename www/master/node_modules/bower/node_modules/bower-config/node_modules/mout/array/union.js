var unique = require('./unique');
var append = require('./append');

    /**
     * Concat multiple arrays and remove duplicates
     */
    function union(arrs) {
        var results = [];
        var i = -1, len = arguments.length;
        while (++i < len) {
            append(results, arguments[i]);
        }

        return unique(results);
    }

    module.exports = union;


