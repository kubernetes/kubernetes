define(['./decode', './getQuery'], function (decode, getQuery) {

    /**
     * Get query string, parses and decodes it.
     */
    function parse(url, shouldTypecast) {
        return decode(getQuery(url), shouldTypecast);
    }

    return parse;
});

