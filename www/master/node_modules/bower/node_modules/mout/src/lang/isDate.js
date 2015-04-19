define(['./isKind'], function (isKind) {
    /**
     */
    function isDate(val) {
        return isKind(val, 'Date');
    }
    return isDate;
});
