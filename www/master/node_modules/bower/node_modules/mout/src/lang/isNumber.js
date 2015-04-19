define(['./isKind'], function (isKind) {
    /**
     */
    function isNumber(val) {
        return isKind(val, 'Number');
    }
    return isNumber;
});
