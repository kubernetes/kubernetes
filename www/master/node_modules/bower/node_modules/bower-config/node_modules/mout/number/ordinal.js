var toInt = require('./toInt');
var nth = require('./nth');

    /**
     * converts number into ordinal form (1st, 2nd, 3rd, 4th, ...)
     */
    function ordinal(n){
       n = toInt(n);
       return n + nth(n);
    }

    module.exports = ordinal;


