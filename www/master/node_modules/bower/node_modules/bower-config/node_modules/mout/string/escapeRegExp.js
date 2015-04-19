var toString = require('../lang/toString');

    /**
     * Escape RegExp string chars.
     */
    function escapeRegExp(str) {
        return toString(str).replace(/\W/g,'\\$&');
    }

    module.exports = escapeRegExp;


