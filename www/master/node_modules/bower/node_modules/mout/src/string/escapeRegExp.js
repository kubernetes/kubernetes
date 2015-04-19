define(['../lang/toString'], function(toString) {

    /**
     * Escape RegExp string chars.
     */
    function escapeRegExp(str) {
        return toString(str).replace(/\W/g,'\\$&');
    }

    return escapeRegExp;

});
