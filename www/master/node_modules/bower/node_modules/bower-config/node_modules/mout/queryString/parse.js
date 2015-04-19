var decode = require('./decode');
var getQuery = require('./getQuery');

    /**
     * Get query string, parses and decodes it.
     */
    function parse(url, shouldTypecast) {
        return decode(getQuery(url), shouldTypecast);
    }

    module.exports = parse;


