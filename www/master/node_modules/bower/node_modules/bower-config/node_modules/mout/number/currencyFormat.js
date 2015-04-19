var toNumber = require('../lang/toNumber');

    /**
     * Converts number into currency format
     */
    function currencyFormat(val, nDecimalDigits, decimalSeparator, thousandsSeparator) {
        val = toNumber(val);
        nDecimalDigits = nDecimalDigits == null? 2 : nDecimalDigits;
        decimalSeparator = decimalSeparator == null? '.' : decimalSeparator;
        thousandsSeparator = thousandsSeparator == null? ',' : thousandsSeparator;

        //can't use enforce precision since it returns a number and we are
        //doing a RegExp over the string
        var fixed = val.toFixed(nDecimalDigits),
            //separate begin [$1], middle [$2] and decimal digits [$4]
            parts = new RegExp('^(-?\\d{1,3})((?:\\d{3})+)(\\.(\\d{'+ nDecimalDigits +'}))?$').exec( fixed );

        if(parts){ //val >= 1000 || val <= -1000
            return parts[1] + parts[2].replace(/\d{3}/g, thousandsSeparator + '$&') + (parts[4] ? decimalSeparator + parts[4] : '');
        }else{
            return fixed.replace('.', decimalSeparator);
        }
    }

    module.exports = currencyFormat;


