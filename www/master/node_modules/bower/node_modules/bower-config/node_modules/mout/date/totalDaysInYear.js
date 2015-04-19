var isLeapYear = require('./isLeapYear');

    /**
     * return the amount of days in the year following the gregorian calendar
     * and leap years
     */
    function totalDaysInYear(fullYear){
        return isLeapYear(fullYear)? 366 : 365;
    }

    module.exports = totalDaysInYear;


