define(['../lang/isDate', './isLeapYear'], function (isDate, isLeapYear) {

    var DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

    /**
     * returns the total amount of days in the month (considering leap years)
     */
    function totalDaysInMonth(fullYear, monthIndex){
        if (isDate(fullYear)) {
            monthIndex = fullYear.getMonth();
        }

        if (monthIndex === 1 && isLeapYear(fullYear)) {
            return 29;
        } else {
            return DAYS_IN_MONTH[monthIndex];
        }
    }

    return totalDaysInMonth;

});
