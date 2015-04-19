var getQuery = require('./getQuery');

    /**
     * Checks if query string contains parameter.
     */
    function contains(url, paramName) {
        var regex = new RegExp('(\\?|&)'+ paramName +'=', 'g'); //matches `?param=` or `&param=`
        return regex.test(getQuery(url));
    }

    module.exports = contains;

