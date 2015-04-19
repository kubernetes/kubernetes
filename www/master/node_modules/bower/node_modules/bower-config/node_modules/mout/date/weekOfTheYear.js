var dayOfTheYear = require('./dayOfTheYear');

    /**
     * Return the week of the year based on given firstDayOfWeek
     */
    function weekOfTheYear(date, firstDayOfWeek){
        firstDayOfWeek = firstDayOfWeek == null? 0 : firstDayOfWeek;
        var doy = dayOfTheYear(date);
        var dow = (7 + date.getDay() - firstDayOfWeek) % 7;
        var relativeWeekDay = 6 - firstDayOfWeek - dow;
        return Math.floor((doy + relativeWeekDay) / 7);
    }

    module.exports = weekOfTheYear;


