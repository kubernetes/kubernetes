var lpad = require('../string/lpad');
var toNumber = require('../lang/toNumber');

    /**
     * Add padding zeros if n.length < minLength.
     */
    function pad(n, minLength, char){
        n = toNumber(n);
        return lpad(''+ n, minLength, char || '0');
    }

    module.exports = pad;


