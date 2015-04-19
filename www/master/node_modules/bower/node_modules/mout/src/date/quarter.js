define(function () {

    /**
     * gets date quarter
     */
    function quarter(date){
        var month = date.getMonth();
        if (month < 3) return 1;
        if (month < 6) return 2;
        if (month < 9) return 3;
        return 4;
    }

    return quarter;

});
