/** 
 * @fileoverview 
 *
 * JsWorld
 *
 * <p>Javascript library for localised formatting and parsing of:
 *    <ul>
 *        <li>Numbers
 *        <li>Dates and times
 *        <li>Currency
 *    </ul>
 *
 * <p>The library classes are configured with standard POSIX locale definitions
 * derived from Unicode's Common Locale Data Repository (CLDR).
 *
 * <p>Website: <a href="http://software.dzhuvinov.com/jsworld.html">JsWorld</a>
 *
 * @author Vladimir Dzhuvinov
 * @version 2.5 (2011-12-23)
 */



/** 
 * @namespace Namespace container for the JsWorld library objects.
 */
jsworld = {};


/** 
 * @function
 * 
 * @description Formats a JavaScript Date object as an ISO-8601 date/time 
 * string.
 *
 * @param {Date} [d] A valid JavaScript Date object. If undefined the 
 *        current date/time will be used.
 * @param {Boolean} [withTZ] Include timezone offset, default false.
 *
 * @returns {String} The date/time formatted as YYYY-MM-DD HH:MM:SS.
 */
jsworld.formatIsoDateTime = function(d, withTZ) {

	if (typeof d === "undefined")
		d = new Date(); // now
	
	if (typeof withTZ === "undefined")
		withTZ = false;
	
	var s = jsworld.formatIsoDate(d) + " " + jsworld.formatIsoTime(d);
	
	if (withTZ) {
	
		var diff = d.getHours() - d.getUTCHours();
		var hourDiff = Math.abs(diff);
		
		var minuteUTC = d.getUTCMinutes();
		var minute = d.getMinutes();
		
		if (minute != minuteUTC && minuteUTC < 30 && diff < 0)
			hourDiff--;
			
		if (minute != minuteUTC && minuteUTC > 30 && diff > 0)
			hourDiff--;
		
		var minuteDiff;
		if (minute != minuteUTC)
			minuteDiff = ":30";
		else
			minuteDiff = ":00";
		
		var timezone;
		if (hourDiff < 10)
			timezone = "0" + hourDiff + minuteDiff;
		
		else
			timezone = "" + hourDiff + minuteDiff;

		if (diff < 0)
			timezone = "-" + timezone;
		
		else
			timezone = "+" + timezone;
		
		s = s + timezone;
	}
	
	return s;
};


/** 
 * @function
 * 
 * @description Formats a JavaScript Date object as an ISO-8601 date string.
 *
 * @param {Date} [d] A valid JavaScript Date object. If undefined the current 
 *        date will be used.
 *
 * @returns {String} The date formatted as YYYY-MM-DD.
 */
jsworld.formatIsoDate = function(d) {

	if (typeof d === "undefined")
		d = new Date(); // now
	
	var year = d.getFullYear();
	var month = d.getMonth() + 1;
	var day = d.getDate();
	
	return year + "-" + jsworld._zeroPad(month, 2) + "-" + jsworld._zeroPad(day, 2);
};


/** 
 * @function
 * 
 * @description Formats a JavaScript Date object as an ISO-8601 time string.
 *
 * @param {Date} [d] A valid JavaScript Date object. If undefined the current 
 *        time will be used.
 *
 * @returns {String} The time formatted as HH:MM:SS.
 */
jsworld.formatIsoTime = function(d) {

	if (typeof d === "undefined")
		d = new Date(); // now
	
	var hour = d.getHours();
	var minute = d.getMinutes();
	var second = d.getSeconds();
	
	return jsworld._zeroPad(hour, 2) + ":" + jsworld._zeroPad(minute, 2) + ":" + jsworld._zeroPad(second, 2);
};


/** 
 * @function
 * 
 * @description Parses an ISO-8601 formatted date/time string to a JavaScript 
 * Date object.
 *
 * @param {String} isoDateTimeVal An ISO-8601 formatted date/time string.
 *
 * <p>Accepted formats:
 *
 * <ul>
 *     <li>YYYY-MM-DD HH:MM:SS
 *     <li>YYYYMMDD HHMMSS
 *     <li>YYYY-MM-DD HHMMSS
 *     <li>YYYYMMDD HH:MM:SS
 * </ul>
 *
 * @returns {Date} The corresponding Date object.
 *
 * @throws Error on a badly formatted date/time string or on a invalid date.
 */
jsworld.parseIsoDateTime = function(isoDateTimeVal) {

	if (typeof isoDateTimeVal != "string")
		throw "Error: The parameter must be a string";

	// First, try to match "YYYY-MM-DD HH:MM:SS" format
	var matches = isoDateTimeVal.match(/^(\d\d\d\d)-(\d\d)-(\d\d)[T ](\d\d):(\d\d):(\d\d)/);
	
	// If unsuccessful, try to match "YYYYMMDD HHMMSS" format
	if (matches === null)
		matches = isoDateTimeVal.match(/^(\d\d\d\d)(\d\d)(\d\d)[T ](\d\d)(\d\d)(\d\d)/);
		
	// ... try to match "YYYY-MM-DD HHMMSS" format
	if (matches === null)
		matches = isoDateTimeVal.match(/^(\d\d\d\d)-(\d\d)-(\d\d)[T ](\d\d)(\d\d)(\d\d)/);
	
	// ... try to match "YYYYMMDD HH:MM:SS" format
	if (matches === null)
		matches = isoDateTimeVal.match(/^(\d\d\d\d)-(\d\d)-(\d\d)[T ](\d\d):(\d\d):(\d\d)/);

	// Report bad date/time string
	if (matches === null)
		throw "Error: Invalid ISO-8601 date/time string";

	// Force base 10 parse int as some values may have leading zeros!
	// (to avoid implicit octal base conversion)
	var year = parseInt(matches[1], 10);
	var month = parseInt(matches[2], 10);
	var day = parseInt(matches[3], 10);
	
	var hour = parseInt(matches[4], 10);
	var mins = parseInt(matches[5], 10);
	var secs = parseInt(matches[6], 10);
	
	// Simple value range check, leap years not checked
	// Note: the originial ISO time spec for leap hours (24:00:00) and seconds (00:00:60) is not supported
	if (month < 1 || month > 12 ||
	    day   < 1 || day   > 31 ||
	    hour  < 0 || hour  > 23 ||
	    mins  < 0 || mins  > 59 ||
	    secs  < 0 || secs  > 59    )
	    
		throw "Error: Invalid ISO-8601 date/time value";

	var d = new Date(year, month - 1, day, hour, mins, secs);
	
	// Check if the input date was valid 
	// (JS Date does automatic forward correction)
	if (d.getDate() != day || d.getMonth() +1 != month)
		throw "Error: Invalid date";
	
	return d;
};


/** 
 * @function
 * 
 * @description Parses an ISO-8601 formatted date string to a JavaScript 
 * Date object.
 *
 * @param {String} isoDateVal An ISO-8601 formatted date string.
 *
 * <p>Accepted formats:
 *
 * <ul>
 *     <li>YYYY-MM-DD
 *     <li>YYYYMMDD
 * </ul>
 *
 * @returns {Date} The corresponding Date object.
 *
 * @throws Error on a badly formatted date string or on a invalid date.
 */
jsworld.parseIsoDate = function(isoDateVal) {

	if (typeof isoDateVal != "string")
		throw "Error: The parameter must be a string";

	// First, try to match "YYYY-MM-DD" format
	var matches = isoDateVal.match(/^(\d\d\d\d)-(\d\d)-(\d\d)/);
	
	// If unsuccessful, try to match "YYYYMMDD" format
	if (matches === null)
		matches = isoDateVal.match(/^(\d\d\d\d)(\d\d)(\d\d)/);

	// Report bad date/time string
	if (matches === null)
		throw "Error: Invalid ISO-8601 date string";

	// Force base 10 parse int as some values may have leading zeros!
	// (to avoid implicit octal base conversion)
	var year = parseInt(matches[1], 10);
	var month = parseInt(matches[2], 10);
	var day = parseInt(matches[3], 10);
	
	// Simple value range check, leap years not checked
	if (month < 1 || month > 12 ||
	    day   < 1 || day   > 31    )
	    
		throw "Error: Invalid ISO-8601 date value";

	var d = new Date(year, month - 1, day);
	
	// Check if the input date was valid 
	// (JS Date does automatic forward correction)
	if (d.getDate() != day || d.getMonth() +1 != month)
		throw "Error: Invalid date";
	
	return d;
};


/** 
 * @function
 * 
 * @description Parses an ISO-8601 formatted time string to a JavaScript 
 * Date object.
 *
 * @param {String} isoTimeVal An ISO-8601 formatted time string.
 *
 * <p>Accepted formats:
 *
 * <ul>
 *     <li>HH:MM:SS
 *     <li>HHMMSS
 * </ul>
 *
 * @returns {Date} The corresponding Date object, with year, month and day set
 *          to zero.
 *
 * @throws Error on a badly formatted time string.
 */
jsworld.parseIsoTime = function(isoTimeVal) {

	if (typeof isoTimeVal != "string")
		throw "Error: The parameter must be a string";

	// First, try to match "HH:MM:SS" format
	var matches = isoTimeVal.match(/^(\d\d):(\d\d):(\d\d)/);
	
	// If unsuccessful, try to match "HHMMSS" format
	if (matches === null)
		matches = isoTimeVal.match(/^(\d\d)(\d\d)(\d\d)/);
	
	// Report bad date/time string
	if (matches === null)
		throw "Error: Invalid ISO-8601 date/time string";

	// Force base 10 parse int as some values may have leading zeros!
	// (to avoid implicit octal base conversion)
	var hour = parseInt(matches[1], 10);
	var mins = parseInt(matches[2], 10);
	var secs = parseInt(matches[3], 10);
	
	// Simple value range check, leap years not checked
	if (hour < 0 || hour > 23 ||
	    mins < 0 || mins > 59 ||
	    secs < 0 || secs > 59    )
	    
		throw "Error: Invalid ISO-8601 time value";

	return new Date(0, 0, 0, hour, mins, secs);
};


/**
 * @private
 *
 * @description Trims leading and trailing whitespace from a string.
 *
 * <p>Used non-regexp the method from http://blog.stevenlevithan.com/archives/faster-trim-javascript
 *
 * @param {String} str The string to trim.
 *
 * @returns {String} The trimmed string.
 */
jsworld._trim = function(str) {

	var whitespace = ' \n\r\t\f\x0b\xa0\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u200b\u2028\u2029\u3000';
	
	for (var i = 0; i < str.length; i++) {
	
		if (whitespace.indexOf(str.charAt(i)) === -1) {
			str = str.substring(i);
			break;
		}
	}
	
	for (i = str.length - 1; i >= 0; i--) {
		if (whitespace.indexOf(str.charAt(i)) === -1) {
			str = str.substring(0, i + 1);
			break;
		}
	}
	
	return whitespace.indexOf(str.charAt(0)) === -1 ? str : '';
};



/**
 * @private
 *
 * @description Returns true if the argument represents a decimal number.
 *
 * @param {Number|String} arg The argument to test.
 *
 * @returns {Boolean} true if the argument represents a decimal number, 
 *          otherwise false.
 */
jsworld._isNumber = function(arg) {

	if (typeof arg == "number")
		return true;
	
	if (typeof arg != "string")
		return false;
	
	// ensure string
	var s = arg + "";
	
	return (/^-?(\d+|\d*\.\d+)$/).test(s);
};


/**
 * @private
 *
 * @description Returns true if the argument represents a decimal integer.
 *
 * @param {Number|String} arg The argument to test.
 *
 * @returns {Boolean} true if the argument represents an integer, otherwise 
 *          false.
 */
jsworld._isInteger = function(arg) {

	if (typeof arg != "number" && typeof arg != "string")
		return false;

	// convert to string
	var s = arg + "";

	return (/^-?\d+$/).test(s);
};


/**
 * @private
 *
 * @description Returns true if the argument represents a decimal float.
 *
 * @param {Number|String} arg The argument to test.
 *
 * @returns {Boolean} true if the argument represents a float, otherwise false.
 */
jsworld._isFloat = function(arg) {

	if (typeof arg != "number" && typeof arg != "string")
		return false;
	
	// convert to string
	var s = arg + "";
	
	return (/^-?\.\d+?$/).test(s);
};


/** 
 * @private
 *
 * @description Checks if the specified formatting option is contained 
 * within the options string.
 * 
 * @param {String} option The option to search for.
 * @param {String} optionsString The options string.
 *
 * @returns {Boolean} true if the flag is found, else false
 */
jsworld._hasOption = function(option, optionsString) {

	if (typeof option != "string" || typeof optionsString != "string")
		return false;

	if (optionsString.indexOf(option) != -1)
		return true;
	else
		return false;
};


/**
 * @private
 *
 * @description String replacement function.
 *
 * @param {String} s The string to work on.
 * @param {String} target The string to search for.
 * @param {String} replacement The replacement.
 *
 * @returns {String} The new string.
 */
jsworld._stringReplaceAll = function(s, target, replacement) {

	var out;

	if (target.length == 1 && replacement.length == 1) {
		// simple char/char case somewhat faster
		out = "";
	
		for (var i = 0; i < s.length; i++) {
			
			if (s.charAt(i) == target.charAt(0))
				out = out + replacement.charAt(0);
			else
				out = out + s.charAt(i);
		}
		
		return out;
	}
	else {
		// longer target and replacement strings
		out = s;

		var index = out.indexOf(target);
		
		while (index != -1) {
		
			out = out.replace(target, replacement);
			
			index = out.indexOf(target);
		}

		return out;
	}
};


/**
 * @private
 *
 * @description Tests if a string starts with the specified substring.
 *
 * @param {String} testedString The string to test.
 * @param {String} sub The string to match.
 *
 * @returns {Boolean} true if the test succeeds.
 */
jsworld._stringStartsWith = function (testedString, sub) {
	
	if (testedString.length < sub.length)
		return false;
	
	for (var i = 0; i < sub.length; i++) {
		if (testedString.charAt(i) != sub.charAt(i))
			return false;
	}
	
	return true;
};


/** 
 * @private
 *
 * @description Gets the requested precision from an options string.
 *
 * <p>Example: ".3" returns 3 decimal places precision.
 *
 * @param {String} optionsString The options string.
 *
 * @returns {integer Number} The requested precision, -1 if not specified.
 */
jsworld._getPrecision = function (optionsString) {

	if (typeof optionsString != "string")
		return -1;

	var m = optionsString.match(/\.(\d)/);
	if (m)
		return parseInt(m[1], 10);
	else
		return -1;
};


/** 
 * @private
 *
 * @description Takes a decimal numeric amount (optionally as string) and 
 * returns its integer and fractional parts packed into an object.
 *
 * @param {Number|String} amount The amount, e.g. "123.45" or "-56.78"
 * 
 * @returns {object} Parsed amount object with properties:
 *         {String} integer  : the integer part
 *         {String} fraction : the fraction part
 */
jsworld._splitNumber = function (amount) {

	if (typeof amount == "number")
		amount = amount + "";

	var obj = {};

	// remove negative sign
	if (amount.charAt(0) == "-")
		amount = amount.substring(1);

	// split amount into integer and decimal parts
	var amountParts = amount.split(".");
	if (!amountParts[1])
		amountParts[1] = ""; // we need "" instead of null

	obj.integer = amountParts[0];
	obj.fraction = amountParts[1];

	return obj;
};


/** 
 * @private
 *
 * @description Formats the integer part using the specified grouping
 * and thousands separator.
 * 
 * @param {String} intPart The integer part of the amount, as string.
 * @param {String} grouping The grouping definition.
 * @param {String} thousandsSep The thousands separator.
 * 
 * @returns {String} The formatted integer part.
 */
jsworld._formatIntegerPart = function (intPart, grouping, thousandsSep) {

	// empty separator string? no grouping?
	// -> return immediately with no formatting!
	if (thousandsSep == "" || grouping == "-1")
		return intPart;

	// turn the semicolon-separated string of integers into an array
	var groupSizes = grouping.split(";");

	// the formatted output string
	var out = "";

	// the intPart string position to process next,
	// start at string end, e.g. "10000000<starts here"
	var pos = intPart.length;

	// process the intPart string backwards
	//     "1000000000"
	//            <---\ direction
	var size;
	
	while (pos > 0) {

		// get next group size (if any, otherwise keep last)
		if (groupSizes.length > 0)
			size = parseInt(groupSizes.shift(), 10);

		// int parse error?
		if (isNaN(size))
			throw "Error: Invalid grouping";

		// size is -1? -> no more grouping, so just copy string remainder
		if (size == -1) {
			out = intPart.substring(0, pos) + out;
			break;
		}

		pos -= size; // move to next sep. char. position

		// position underrun? -> just copy string remainder
		if (pos < 1) {
			out = intPart.substring(0, pos + size) + out;
			break;
		}

		// extract group and apply sep. char.
		out = thousandsSep + intPart.substring(pos, pos + size) + out;
	}

	return out;
};
	
	
/** 
 * @private
 *
 * @description Formats the fractional part to the specified decimal 
 * precision.
 *
 * @param {String} fracPart The fractional part of the amount
 * @param {integer Number} precision The desired decimal precision
 *
 * @returns {String} The formatted fractional part.
 */
jsworld._formatFractionPart = function (fracPart, precision) {

	// append zeroes up to precision if necessary
	for (var i=0; fracPart.length < precision; i++)
		fracPart = fracPart + "0";

	return fracPart;
};


/** 
 * @private 
 *
 * @desription Converts a number to string and pad it with leading zeroes if the
 * string is shorter than length.
 *
 * @param {integer Number} number The number value subjected to selective padding.
 * @param {integer Number} length If the number has fewer digits than this length
 *        apply padding.
 *
 * @returns {String} The formatted string.
 */
jsworld._zeroPad = function(number, length) {

	// ensure string
	var s = number + "";

	while (s.length < length)
		s = "0" + s;
	
	return s;
};


/** 
 * @private 
 * @description Converts a number to string and pads it with leading spaces if 
 * the string is shorter than length.
 *
 * @param {integer Number} number The number value subjected to selective padding.
 * @param {integer Number} length If the number has fewer digits than this length
 *        apply padding.
 *
 * @returns {String} The formatted string.
 */
jsworld._spacePad = function(number, length) {

	// ensure string
	var s = number + "";

	while (s.length < length)
		s = " " + s;
	
	return s;
};



/**
 * @class
 * Represents a POSIX-style locale with its numeric, monetary and date/time 
 * properties. Also provides a set of locale helper methods.
 *
 * <p>The locale properties follow the POSIX standards:
 *
 * <ul>
 *     <li><a href="http://www.opengroup.org/onlinepubs/000095399/basedefs/xbd_chap07.html#tag_07_03_04">POSIX LC_NUMERIC</a>
 *     <li><a href="http://www.opengroup.org/onlinepubs/009695399/basedefs/xbd_chap07.html#tag_07_03_03">POSIX LC_MONETARY</a>
 *     <li><a href="http://www.opengroup.org/onlinepubs/000095399/basedefs/xbd_chap07.html#tag_07_03_05">POSIX LC_TIME</a>
 * </ul>
 *
 * @public
 * @constructor
 * @description Creates a new locale object (POSIX-style) with the specified
 * properties.
 *
 * @param {object} properties An object containing the raw locale properties:
 *
 *        @param {String} properties.decimal_point
 *
 *        A string containing the symbol that shall be used as the decimal
 *        delimiter (radix character) in numeric, non-monetary formatted
 *        quantities. This property cannot be omitted and cannot be set to the
 *        empty string.
 *
 *
 *        @param {String} properties.thousands_sep
 *
 *        A string containing the symbol that shall be used as a separator for
 *        groups of digits to the left of the decimal delimiter in numeric,
 *        non-monetary formatted monetary quantities.
 *
 *
 *        @param {String} properties.grouping
 *
 *        Defines the size of each group of digits in formatted non-monetary
 *        quantities. The operand is a sequence of integers separated by
 *        semicolons. Each integer specifies the number of digits in each group,
 *        with the initial integer defining the size of the group immediately
 *        preceding the decimal delimiter, and the following integers defining
 *        the preceding groups. If the last integer is not -1, then the size of
 *        the previous group (if any) shall be repeatedly used for the
 *        remainder of the digits. If the last integer is -1, then no further
 *        grouping shall be performed.
 *
 *
 *        @param {String} properties.int_curr_symbol
 *
 *        The first three letters signify the ISO-4217 currency code,
 *        the fourth letter is the international symbol separation character
 *        (normally a space).
 *
 *
 *        @param {String} properties.currency_symbol
 *
 *        The local shorthand currency symbol, e.g. "$" for the en_US locale
 *
 *
 *        @param {String} properties.mon_decimal_point
 *
 *        The symbol to be used as the decimal delimiter (radix character)
 *
 *
 *        @param {String} properties.mon_thousands_sep
 *
 *        The symbol to be used as a separator for groups of digits to the
 *        left of the decimal delimiter.
 *
 *
 *        @param {String} properties.mon_grouping
 *
 *        A string that defines the size of each group of digits. The
 *        operand is a sequence of integers separated by semicolons (";").
 *        Each integer specifies the number of digits in each group, with the
 *        initial integer defining the size of the group preceding the
 *        decimal delimiter, and the following integers defining the
 *        preceding groups. If the last integer is not -1, then the size of
 *        the previous group (if any) must be repeatedly used for the
 *        remainder of the digits. If the last integer is -1, then no
 *        further grouping is to be performed.
 *
 *
 *        @param {String} properties.positive_sign
 *
 *        The string to indicate a non-negative monetary amount.
 *
 *
 *        @param {String} properties.negative_sign
 *
 *        The string to indicate a negative monetary amount.
 *
 *
 *        @param {integer Number} properties.frac_digits
 *
 *        An integer representing the number of fractional digits (those to
 *        the right of the decimal delimiter) to be written in a formatted
 *        monetary quantity using currency_symbol.
 *
 *
 *        @param {integer Number} properties.int_frac_digits
 *
 *        An integer representing the number of fractional digits (those to
 *        the right of the decimal delimiter) to be written in a formatted
 *        monetary quantity using int_curr_symbol.
 *
 *
 *        @param {integer Number} properties.p_cs_precedes
 *
 *        An integer set to 1 if the currency_symbol precedes the value for a
 *        monetary quantity with a non-negative value, and set to 0 if the
 *        symbol succeeds the value.
 *
 *
 *        @param {integer Number} properties.n_cs_precedes
 *
 *        An integer set to 1 if the currency_symbol precedes the value for a
 *        monetary quantity with a negative value, and set to 0 if the symbol
 *        succeeds the value.
 *
 *
 *        @param {integer Number} properties.p_sep_by_space
 *
 *        Set to a value indicating the separation of the currency_symbol,
 *        the sign string, and the value for a non-negative formatted monetary
 *        quantity:
 *        
 *             <p>0 No space separates the currency symbol and value.</p>
 *
 *             <p>1 If the currency symbol and sign string are adjacent, a space
 *                  separates them from the value; otherwise, a space separates
 *                  the currency symbol from the value.</p>
 *
 *             <p>2 If the currency symbol and sign string are adjacent, a space
 *                  separates them; otherwise, a space separates the sign string
 *                  from the value.</p>
 *
 *
 *        @param {integer Number} properties.n_sep_by_space
 *
 *        Set to a value indicating the separation of the currency_symbol,
 *        the sign string, and the value for a negative formatted monetary
 *        quantity. Rules same as for p_sep_by_space.
 *
 *
 *        @param {integer Number} properties.p_sign_posn
 *
 *        An integer set to a value indicating the positioning of the
 *        positive_sign for a monetary quantity with a non-negative value:
 *	
 *	       <p>0 Parentheses enclose the quantity and the currency_symbol.</p>
 *
 *	       <p>1 The sign string precedes the quantity and the currency_symbol.</p>
 *
 *	       <p>2 The sign string succeeds the quantity and the currency_symbol.</p>
 *
 *	       <p>3 The sign string precedes the currency_symbol.</p>
 *
 *	       <p>4 The sign string succeeds the currency_symbol.</p>
 *
 *
 *	  @param {integer Number} properties.n_sign_posn
 *
 *	  An integer set to a value indicating the positioning of the
 *	  negative_sign for a negative formatted monetary quantity. Rules same
 *	  as for p_sign_posn.
 *
 *
 *	  @param {integer Number} properties.int_p_cs_precedes
 *
 *	  An integer set to 1 if the int_curr_symbol precedes the value for a
 *	  monetary quantity with a non-negative value, and set to 0 if the
 *	  symbol succeeds the value.
 *
 *
 *	  @param {integer Number} properties.int_n_cs_precedes
 *
 *	  An integer set to 1 if the int_curr_symbol precedes the value for a
 *	  monetary quantity with a negative value, and set to 0 if the symbol
 *	  succeeds the value.
 *
 *
 *	  @param {integer Number} properties.int_p_sep_by_space
 *
 *	  Set to a value indicating the separation of the int_curr_symbol,
 *	  the sign string, and the value for a non-negative internationally
 *	  formatted monetary quantity. Rules same as for p_sep_by_space.
 *
 *
 *	  @param {integer Number} properties.int_n_sep_by_space
 *
 *	  Set to a value indicating the separation of the int_curr_symbol,
 *	  the sign string, and the value for a negative internationally
 *	  formatted monetary quantity. Rules same as for p_sep_by_space.
 *
 *
 *	  @param {integer Number} properties.int_p_sign_posn
 *
 *	  An integer set to a value indicating the positioning of the
 *	  positive_sign for a positive monetary quantity formatted with the
 *	  international format. Rules same as for p_sign_posn.
 *
 *
 *	  @param {integer Number} properties.int_n_sign_posn
 *
 *	  An integer set to a value indicating the positioning of the
 *	  negative_sign for a negative monetary quantity formatted with the
 *	  international format. Rules same as for p_sign_posn.
 *
 *
 *        @param {String[] | String} properties.abday
 *
 *        The abbreviated weekday names, corresponding to the %a conversion
 *        specification. The property must be either an array of 7 strings or
 *        a string consisting of 7 semicolon-separated substrings, each 
 *        surrounded by double-quotes. The first must be the abbreviated name 
 *        of the day corresponding to Sunday, the second the abbreviated name 
 *        of the day corresponding to Monday, and so on.
 *        
 *
 *        @param {String[] | String} properties.day
 *
 *        The full weekday names, corresponding to the %A conversion
 *        specification. The property must be either an array of 7 strings or
 *        a string consisting of 7 semicolon-separated substrings, each 
 *        surrounded by double-quotes. The first must be the full name of the 
 *        day corresponding to Sunday, the second the full name of the day 
 *        corresponding to Monday, and so on.
 *        
 *
 *        @param {String[] | String} properties.abmon
 *
 *        The abbreviated month names, corresponding to the %b conversion
 *        specification. The property must be either an array of 12 strings or
 *        a string consisting of 12 semicolon-separated substrings, each 
 *        surrounded by double-quotes. The first must be the abbreviated name 
 *        of the first month of the year (January), the second the abbreviated 
 *        name of the second month, and so on.
 *        
 *
 *        @param {String[] | String} properties.mon
 *
 *        The full month names, corresponding to the %B conversion
 *        specification. The property must be either an array of 12 strings or
 *        a string consisting of 12 semicolon-separated substrings, each 
 *        surrounded by double-quotes. The first must be the full name of the 
 *        first month of the year (January), the second the full name of the second 
 *        month, and so on.
 *        
 *
 *        @param {String} properties.d_fmt
 *
 *        The appropriate date representation. The string may contain any
 *        combination of characters and conversion specifications (%<char>).
 *        
 *
 *        @param {String} properties.t_fmt
 *
 *        The appropriate time representation. The string may contain any
 *        combination of characters and conversion specifications (%<char>).
 *        
 *
 *        @param {String} properties.d_t_fmt
 *
 *        The appropriate date and time representation. The string may contain
 *        any combination of characters and conversion specifications (%<char>).
 *
 *
 *        @param {String[] | String} properties.am_pm
 *
 *        The appropriate representation of the ante-meridiem and post-meridiem
 *        strings, corresponding to the %p conversion specification. The property 
 *        must be either an array of 2 strings or a string consisting of 2 
 *        semicolon-separated substrings, each surrounded by double-quotes. 
 *        The first string must represent the ante-meridiem designation, the 
 *        last string the post-meridiem designation.
 *
 *
 * @throws @throws Error on a undefined or invalid locale property.
 */
jsworld.Locale = function(properties) {
	
	
	/**
	 * @private
	 *
	 * @description Identifies the class for internal library purposes.
	 */
	this._className = "jsworld.Locale";
	
	
	/** 
	 * @private 
	 *
	 * @description Parses a day or month name definition list, which
	 * could be a ready JS array, e.g. ["Mon", "Tue", "Wed"...] or
	 * it could be a string formatted according to the classic POSIX
	 * definition e.g. "Mon";"Tue";"Wed";...
	 *
	 * @param {String[] | String} namesAn array or string defining 
	 *        the week/month names.
	 * @param {integer Number} expectedItems The number of expected list
	 *        items, e.g. 7 for weekdays, 12 for months.
	 *
	 * @returns {String[]} The parsed (and checked) items.
	 * 
	 * @throws Error on missing definition, unexpected item count or
	 *         missing double-quotes.
	 */
	this._parseList = function(names, expectedItems) {
		
		var array = [];
		
		if (names == null) {
			throw "Names not defined";
		}
		else if (typeof names == "object") {
			// we got a ready array
			array = names;
		}
		else if (typeof names == "string") {
			// we got the names in the classic POSIX form, do parse
			array = names.split(";", expectedItems);
		
			for (var i = 0; i < array.length; i++) {
				// check for and strip double quotes
				if (array[i][0] == "\"" && array[i][array[i].length - 1] == "\"")
					array[i] = array[i].slice(1, -1);
				else
					throw "Missing double quotes";
			}
		}
		else {
			throw "Names must be an array or a string";
		}
		
		if (array.length != expectedItems)
			throw "Expected " + expectedItems + " items, got " + array.length;
		
		return array;
	};
	
	
	/**
	 * @private
	 *
	 * @description Validates a date/time format string, such as "H:%M:%S". 
	 * Checks that the argument is of type "string" and is not empty.
	 *
	 * @param {String} formatString The format string.
	 *
	 * @returns {String} The validated string.
	 *
	 * @throws Error on null or empty string.
	 */
	this._validateFormatString = function(formatString) {
		
		if (typeof formatString == "string" && formatString.length > 0)
			return formatString;
		else
			throw "Empty or no string";
	};
	
	
	// LC_NUMERIC

	if (properties == null || typeof properties != "object")
		throw "Error: Invalid/missing locale properties";
	
	
	if (typeof properties.decimal_point != "string")
		throw "Error: Invalid/missing decimal_point property";
	
	this.decimal_point = properties.decimal_point;
	
	
	if (typeof properties.thousands_sep != "string")
		throw "Error: Invalid/missing thousands_sep property";
	
	this.thousands_sep = properties.thousands_sep;
	
	
	if (typeof properties.grouping != "string")
		throw "Error: Invalid/missing grouping property";
	
	this.grouping = properties.grouping;
	
	
	// LC_MONETARY
	
	if (typeof properties.int_curr_symbol != "string")
		throw "Error: Invalid/missing int_curr_symbol property";
	
	if (! /[A-Za-z]{3}.?/.test(properties.int_curr_symbol))
		throw "Error: Invalid int_curr_symbol property";
	
	this.int_curr_symbol = properties.int_curr_symbol;
	

	if (typeof properties.currency_symbol != "string")
		throw "Error: Invalid/missing currency_symbol property";
	
	this.currency_symbol = properties.currency_symbol;
	
	
	if (typeof properties.frac_digits != "number" && properties.frac_digits < 0)
		throw "Error: Invalid/missing frac_digits property";
	
	this.frac_digits = properties.frac_digits;
	
	
	// may be empty string/null for currencies with no fractional part
	if (properties.mon_decimal_point === null || properties.mon_decimal_point == "") {
	
		if (this.frac_digits > 0)
			throw "Error: Undefined mon_decimal_point property";
		else
			properties.mon_decimal_point = "";
	}
	
	if (typeof properties.mon_decimal_point != "string")
		throw "Error: Invalid/missing mon_decimal_point property";
	
	this.mon_decimal_point = properties.mon_decimal_point;
	
	
	if (typeof properties.mon_thousands_sep != "string")
		throw "Error: Invalid/missing mon_thousands_sep property";
	
	this.mon_thousands_sep = properties.mon_thousands_sep;
	
	
	if (typeof properties.mon_grouping != "string")
		throw "Error: Invalid/missing mon_grouping property";
	
	this.mon_grouping = properties.mon_grouping;
	
	
	if (typeof properties.positive_sign != "string")
		throw "Error: Invalid/missing positive_sign property";
	
	this.positive_sign = properties.positive_sign;
	
	
	if (typeof properties.negative_sign != "string")
		throw "Error: Invalid/missing negative_sign property";
	
	this.negative_sign = properties.negative_sign;
	
	
	
	if (properties.p_cs_precedes !== 0 && properties.p_cs_precedes !== 1)
		throw "Error: Invalid/missing p_cs_precedes property, must be 0 or 1";
	
	this.p_cs_precedes = properties.p_cs_precedes;
	
	
	if (properties.n_cs_precedes !== 0 && properties.n_cs_precedes !== 1)
		throw "Error: Invalid/missing n_cs_precedes, must be 0 or 1";
	
	this.n_cs_precedes = properties.n_cs_precedes;
	

	if (properties.p_sep_by_space !== 0 &&
	    properties.p_sep_by_space !== 1 &&
	    properties.p_sep_by_space !== 2)
		throw "Error: Invalid/missing p_sep_by_space property, must be 0, 1 or 2";
	
	this.p_sep_by_space = properties.p_sep_by_space;
	

	if (properties.n_sep_by_space !== 0 &&
	    properties.n_sep_by_space !== 1 &&
	    properties.n_sep_by_space !== 2)
		throw "Error: Invalid/missing n_sep_by_space property, must be 0, 1, or 2";
	
	this.n_sep_by_space = properties.n_sep_by_space;
	

	if (properties.p_sign_posn !== 0 &&
	    properties.p_sign_posn !== 1 &&
	    properties.p_sign_posn !== 2 &&
	    properties.p_sign_posn !== 3 &&
	    properties.p_sign_posn !== 4)
		throw "Error: Invalid/missing p_sign_posn property, must be 0, 1, 2, 3 or 4";
	
	this.p_sign_posn = properties.p_sign_posn;


	if (properties.n_sign_posn !== 0 &&
	    properties.n_sign_posn !== 1 &&
	    properties.n_sign_posn !== 2 &&
	    properties.n_sign_posn !== 3 &&
	    properties.n_sign_posn !== 4)
		throw "Error: Invalid/missing n_sign_posn property, must be 0, 1, 2, 3 or 4";
	
	this.n_sign_posn = properties.n_sign_posn;


	if (typeof properties.int_frac_digits != "number" && properties.int_frac_digits < 0)
		throw "Error: Invalid/missing int_frac_digits property";

	this.int_frac_digits = properties.int_frac_digits;
	
	
	if (properties.int_p_cs_precedes !== 0 && properties.int_p_cs_precedes !== 1)
		throw "Error: Invalid/missing int_p_cs_precedes property, must be 0 or 1";
	
	this.int_p_cs_precedes = properties.int_p_cs_precedes;
	
	
	if (properties.int_n_cs_precedes !== 0 && properties.int_n_cs_precedes !== 1)
		throw "Error: Invalid/missing int_n_cs_precedes property, must be 0 or 1";
	
	this.int_n_cs_precedes = properties.int_n_cs_precedes;
	

	if (properties.int_p_sep_by_space !== 0 &&
	    properties.int_p_sep_by_space !== 1 &&
	    properties.int_p_sep_by_space !== 2)
		throw "Error: Invalid/missing int_p_sep_by_spacev, must be 0, 1 or 2";
		
	this.int_p_sep_by_space = properties.int_p_sep_by_space;


	if (properties.int_n_sep_by_space !== 0 &&
	    properties.int_n_sep_by_space !== 1 &&
	    properties.int_n_sep_by_space !== 2)
		throw "Error: Invalid/missing int_n_sep_by_space property, must be 0, 1, or 2";
	
	this.int_n_sep_by_space = properties.int_n_sep_by_space;
	

	if (properties.int_p_sign_posn !== 0 &&
	    properties.int_p_sign_posn !== 1 &&
	    properties.int_p_sign_posn !== 2 &&
	    properties.int_p_sign_posn !== 3 &&
	    properties.int_p_sign_posn !== 4)
		throw "Error: Invalid/missing int_p_sign_posn property, must be 0, 1, 2, 3 or 4";
	
	this.int_p_sign_posn = properties.int_p_sign_posn;
	
	
	if (properties.int_n_sign_posn !== 0 &&
	    properties.int_n_sign_posn !== 1 &&
	    properties.int_n_sign_posn !== 2 &&
	    properties.int_n_sign_posn !== 3 &&
	    properties.int_n_sign_posn !== 4)
		throw "Error: Invalid/missing int_n_sign_posn property, must be 0, 1, 2, 3 or 4";

	this.int_n_sign_posn = properties.int_n_sign_posn;
	
	
	// LC_TIME
	
	if (properties == null || typeof properties != "object")
		throw "Error: Invalid/missing time locale properties";
	
	
	// parse the supported POSIX LC_TIME properties
	
	// abday
	try  {
		this.abday = this._parseList(properties.abday, 7);
	}
	catch (error) {
		throw "Error: Invalid abday property: " + error;
	}
	
	// day
	try {
		this.day = this._parseList(properties.day, 7);
	}
	catch (error) {
		throw "Error: Invalid day property: " + error;
	}
	
	// abmon
	try {
		this.abmon = this._parseList(properties.abmon, 12);
	} catch (error) {
		throw "Error: Invalid abmon property: " + error;
	}
	
	// mon
	try {
		this.mon = this._parseList(properties.mon, 12);
	} catch (error) {
		throw "Error: Invalid mon property: " + error;
	}
	
	// d_fmt
	try {
		this.d_fmt = this._validateFormatString(properties.d_fmt);
	} catch (error) {
		throw "Error: Invalid d_fmt property: " + error;
	}
	
	// t_fmt
	try {
		this.t_fmt = this._validateFormatString(properties.t_fmt);
	} catch (error) {
		throw "Error: Invalid t_fmt property: " + error;
	}
	
	// d_t_fmt
	try {
		this.d_t_fmt = this._validateFormatString(properties.d_t_fmt);
	} catch (error) {
		throw "Error: Invalid d_t_fmt property: " + error;
	}
	
	// am_pm
	try {
		var am_pm_strings = this._parseList(properties.am_pm, 2);
		this.am = am_pm_strings[0];
		this.pm = am_pm_strings[1];
	} catch (error) {
		// ignore empty/null string errors
		this.am = "";
		this.pm = "";
	}
	
	
	/**
	 * @public
	 *
	 * @description Returns the abbreviated name of the specified weekday.
	 *
	 * @param {integer Number} [weekdayNum] An integer between 0 and 6. Zero 
	 *        corresponds to Sunday, one to Monday, etc. If omitted the
	 *        method will return an array of all abbreviated weekday 
	 *        names.
	 *
	 * @returns {String | String[]} The abbreviated name of the specified weekday
	 *          or an array of all abbreviated weekday names.
	 *
	 * @throws Error on invalid argument.
	 */
	this.getAbbreviatedWeekdayName = function(weekdayNum) {
	
		if (typeof weekdayNum == "undefined" || weekdayNum === null)
			return this.abday;
		
		if (! jsworld._isInteger(weekdayNum) || weekdayNum < 0 || weekdayNum > 6)
			throw "Error: Invalid weekday argument, must be an integer [0..6]";
			
		return this.abday[weekdayNum];
	};
	
	
	/**
	 * @public
	 *
	 * @description Returns the name of the specified weekday.
	 *
	 * @param {integer Number} [weekdayNum] An integer between 0 and 6. Zero 
	 *        corresponds to Sunday, one to Monday, etc. If omitted the
	 *        method will return an array of all weekday names.
	 *
	 * @returns {String | String[]} The name of the specified weekday or an 
	 *          array of all weekday names.
	 *
	 * @throws Error on invalid argument.
	 */
	this.getWeekdayName = function(weekdayNum) {
		
		if (typeof weekdayNum == "undefined" || weekdayNum === null)
			return this.day;
		
		if (! jsworld._isInteger(weekdayNum) || weekdayNum < 0 || weekdayNum > 6)
			throw "Error: Invalid weekday argument, must be an integer [0..6]";
			
		return this.day[weekdayNum];
	};
	
	
	/**
	 * @public
	 *
	 * @description Returns the abbreviated name of the specified month.
	 *
	 * @param {integer Number} [monthNum] An integer between 0 and 11. Zero 
	 *        corresponds to January, one to February, etc. If omitted the
	 *        method will return an array of all abbreviated month names.
	 *
	 * @returns {String | String[]} The abbreviated name of the specified month
	 *          or an array of all abbreviated month names.
	 *
	 * @throws Error on invalid argument.
	 */
	this.getAbbreviatedMonthName = function(monthNum) {
	
		if (typeof monthNum == "undefined" || monthNum === null)
			return this.abmon;
		
		if (! jsworld._isInteger(monthNum) || monthNum < 0 || monthNum > 11)
			throw "Error: Invalid month argument, must be an integer [0..11]";
		
		return this.abmon[monthNum];
	};
	
	
	/**
	 * @public
	 *
	 * @description Returns the name of the specified month.
	 *
	 * @param {integer Number} [monthNum] An integer between 0 and 11. Zero 
	 *        corresponds to January, one to February, etc. If omitted the
	 *        method will return an array of all month names.
	 *
	 * @returns {String | String[]} The name of the specified month or an array 
	 *          of all month names.
	 *
	 * @throws Error on invalid argument.
	 */
	this.getMonthName = function(monthNum) {
	
		if (typeof monthNum == "undefined" || monthNum === null)
			return this.mon;
		
		if (! jsworld._isInteger(monthNum) || monthNum < 0 || monthNum > 11)
			throw "Error: Invalid month argument, must be an integer [0..11]";
		
		return this.mon[monthNum];
	};
	
	
	
	/** 
	 * @public
	 *
	 * @description Gets the decimal delimiter (radix) character for
	 * numeric quantities.
	 *
	 * @returns {String} The radix character.
	 */
	this.getDecimalPoint = function() {
		
		return this.decimal_point;
	};
	
	
	/** 
	 * @public
	 *
	 * @description Gets the local shorthand currency symbol.
	 *
	 * @returns {String} The currency symbol.
	 */
	this.getCurrencySymbol = function() {
		
		return this.currency_symbol;
	};
	
	
	/**
	 * @public
	 *
	 * @description Gets the internaltion currency symbol (ISO-4217 code).
	 *
	 * @returns {String} The international currency symbol.
	 */
	this.getIntCurrencySymbol = function() {
	
		return this.int_curr_symbol.substring(0,3);
	};
	
	
	/** 
	 * @public
	 *
	 * @description Gets the position of the local (shorthand) currency 
	 * symbol relative to the amount. Assumes a non-negative amount.
	 *
	 * @returns {Boolean} True if the symbol precedes the amount, false if
	 * the symbol succeeds the amount.
	 */
	this.currencySymbolPrecedes = function() {
		
		if (this.p_cs_precedes == 1)
			return true;
		else
			return false;
	};
	
	
	/** 
	 * @public
	 *
	 * @description Gets the position of the international (ISO-4217 code) 
	 * currency symbol relative to the amount. Assumes a non-negative 
	 * amount.
	 *
	 * @returns {Boolean} True if the symbol precedes the amount, false if
	 * the symbol succeeds the amount.
	 */
	this.intCurrencySymbolPrecedes = function() {
		
		if (this.int_p_cs_precedes == 1)
			return true;
		else
			return false;

	};
	
	
	/** 
	 * @public
	 *
	 * @description Gets the decimal delimiter (radix) for monetary
	 * quantities.
	 *
	 * @returns {String} The radix character.
	 */
	this.getMonetaryDecimalPoint = function() {
		
		return this.mon_decimal_point;
	};
	
	
	/** 
	 * @public
	 *
	 * @description Gets the number of fractional digits for local
	 * (shorthand) symbol formatting.
	 *
	 * @returns {integer Number} The number of fractional digits.
	 */
	this.getFractionalDigits = function() {
		
		return this.frac_digits;
	};
	
	
	/** 
	 * @public
	 *
	 * @description Gets the number of fractional digits for
	 * international (ISO-4217 code) formatting.
	 *
	 * @returns {integer Number} The number of fractional digits.
	 */
	this.getIntFractionalDigits = function() {
		
		return this.int_frac_digits;
	};
};



/** 
 * @class 
 * Class for localised formatting of numbers.
 *
 * <p>See: <a href="http://www.opengroup.org/onlinepubs/000095399/basedefs/xbd_chap07.html#tag_07_03_04">
 * POSIX LC_NUMERIC</a>.
 *
 *
 * @public
 * @constructor 
 * @description Creates a new numeric formatter for the specified locale.
 *
 * @param {jsworld.Locale} locale A locale object specifying the required 
 *        POSIX LC_NUMERIC formatting properties.
 *
 * @throws Error on constructor failure.
 */
jsworld.NumericFormatter = function(locale) {

	if (typeof locale != "object" || locale._className != "jsworld.Locale")
		throw "Constructor error: You must provide a valid jsworld.Locale instance";
	
	this.lc = locale;
	
	
	/** 
	 * @public
	 * 
	 * @description Formats a decimal numeric value according to the preset
	 * locale.
	 *
	 * @param {Number|String} number The number to format.
	 * @param {String} [options] Options to modify the formatted output:
	 *        <ul>
	 *            <li>"^"  suppress grouping
	 *            <li>"+"  force positive sign for positive amounts
	 *            <li>"~"  suppress positive/negative sign
	 *            <li>".n" specify decimal precision 'n'
	 *        </ul>
	 *
	 * @returns {String} The formatted number.
	 *
	 * @throws "Error: Invalid input" on bad input.
	 */
	this.format = function(number, options) {
		
		if (typeof number == "string")
			number = jsworld._trim(number);
		
		if (! jsworld._isNumber(number))
			throw "Error: The input is not a number";
		
		var floatAmount = parseFloat(number, 10);
		
		// get the required precision
		var reqPrecision = jsworld._getPrecision(options);
		
		// round to required precision
		if (reqPrecision != -1)
			floatAmount = Math.round(floatAmount * Math.pow(10, reqPrecision)) / Math.pow(10, reqPrecision);
		
		
		// convert the float number to string and parse into
		// object with properties integer and fraction
		var parsedAmount = jsworld._splitNumber(String(floatAmount));
		
		// format integer part with grouping chars
		var formattedIntegerPart;
		
		if (floatAmount === 0)
			formattedIntegerPart = "0";
		else
			formattedIntegerPart = jsworld._hasOption("^", options) ?
				parsedAmount.integer :
				jsworld._formatIntegerPart(parsedAmount.integer, 
				                           this.lc.grouping, 
							   this.lc.thousands_sep);
		
		// format the fractional part
		var formattedFractionPart =
			reqPrecision != -1 ?
			jsworld._formatFractionPart(parsedAmount.fraction, reqPrecision) :
			parsedAmount.fraction;
		
		
		// join the integer and fraction parts using the decimal_point property
		var formattedAmount =
			formattedFractionPart.length ?
			formattedIntegerPart + this.lc.decimal_point + formattedFractionPart :
			formattedIntegerPart;
		
		// prepend sign?
		if (jsworld._hasOption("~", options) || floatAmount === 0) {
			// suppress both '+' and '-' signs, i.e. return abs value
			return formattedAmount; 
		}
		else {
			if (jsworld._hasOption("+", options) || floatAmount < 0) {
				if (floatAmount > 0)
					// force '+' sign for positive amounts
					return "+" + formattedAmount;
				else if (floatAmount < 0)
					// prepend '-' sign
					return "-" + formattedAmount;
				else
					// zero case
					return formattedAmount;
			}
			else {
				// positive amount with no '+' sign
				return formattedAmount;
			}
		}
	};
};


/** 
 * @class 
 * Class for localised formatting of dates and times.
 *
 * <p>See: <a href="http://www.opengroup.org/onlinepubs/000095399/basedefs/xbd_chap07.html#tag_07_03_05">
 * POSIX LC_TIME</a>.
 *
 * @public
 * @constructor
 * @description Creates a new date/time formatter for the specified locale.
 *
 * @param {jsworld.Locale} locale A locale object specifying the required 
 *        POSIX LC_TIME formatting properties.
 *
 * @throws Error on constructor failure.
 */
jsworld.DateTimeFormatter = function(locale) {
	
		
	if (typeof locale != "object" || locale._className != "jsworld.Locale")
		throw "Constructor error: You must provide a valid jsworld.Locale instance.";
	
	this.lc = locale;

	
	/** 
	 * @public 
	 *
	 * @description Formats a date according to the preset locale.
	 *
	 * @param {Date|String} date A valid Date object instance or a string
	 *        containing a valid ISO-8601 formatted date, e.g. "2010-31-03" 
	 *        or "2010-03-31 23:59:59".
	 *
	 * @returns {String} The formatted date
	 *
	 * @throws Error on invalid date argument
	 */
	this.formatDate = function(date) {
		
		var d = null;
		
		if (typeof date == "string") {
			// assume ISO-8601 date string
			try {
				d = jsworld.parseIsoDate(date);
			} catch (error) {
				// try full ISO-8601 date/time string
				d = jsworld.parseIsoDateTime(date);
			}
		}
		else if (date !== null && typeof date == "object") {
			// assume ready Date object
			d = date;
		}
		else {
			throw "Error: Invalid date argument, must be a Date object or an ISO-8601 date/time string";
		}
		
		return this._applyFormatting(d, this.lc.d_fmt);
	};
	
	
	/** 
	 * @public 
	 *
	 * @description Formats a time according to the preset locale.
	 *
	 * @param {Date|String} date A valid Date object instance or a string
	 *        containing a valid ISO-8601 formatted time, e.g. "23:59:59"
	 *        or "2010-03-31 23:59:59".
	 *
	 * @returns {String} The formatted time.
	 *
	 * @throws Error on invalid date argument.
	 */
	this.formatTime = function(date) {
		
		var d = null;
		
		if (typeof date == "string") {
			// assume ISO-8601 time string
			try {
				d = jsworld.parseIsoTime(date);
			} catch (error) {
				// try full ISO-8601 date/time string
				d = jsworld.parseIsoDateTime(date);
			}
		}
		else if (date !== null && typeof date == "object") {
			// assume ready Date object
			d = date;
		}
		else {
			throw "Error: Invalid date argument, must be a Date object or an ISO-8601 date/time string";
		}
		
		return this._applyFormatting(d, this.lc.t_fmt);
	};
	
	
	/** 
	 * @public 
	 *
	 * @description Formats a date/time value according to the preset 
	 * locale.
	 *
	 * @param {Date|String} date A valid Date object instance or a string
	 *        containing a valid ISO-8601 formatted date/time, e.g.
	 *        "2010-03-31 23:59:59".
	 *
	 * @returns {String} The formatted time.
	 *
	 * @throws Error on invalid argument.
	 */
	this.formatDateTime = function(date) {
		
		var d = null;
		
		if (typeof date == "string") {
			// assume ISO-8601 format
			d = jsworld.parseIsoDateTime(date);
		}
		else if (date !== null && typeof date == "object") {
			// assume ready Date object
			d = date;
		}
		else {
			throw "Error: Invalid date argument, must be a Date object or an ISO-8601 date/time string";
		}
		
		return this._applyFormatting(d, this.lc.d_t_fmt);
	};
	
	
	/** 
	 * @private 
	 *
	 * @description Apples formatting to the Date object according to the
	 * format string.
	 *
	 * @param {Date} d A valid Date instance.
	 * @param {String} s The formatting string with '%' placeholders.
	 *
	 * @returns {String} The formatted string.
	 */
	this._applyFormatting = function(d, s) {
		
		s = s.replace(/%%/g, '%');
		s = s.replace(/%a/g, this.lc.abday[d.getDay()]);
		s = s.replace(/%A/g, this.lc.day[d.getDay()]);
		s = s.replace(/%b/g, this.lc.abmon[d.getMonth()]);
		s = s.replace(/%B/g, this.lc.mon[d.getMonth()]);
		s = s.replace(/%d/g, jsworld._zeroPad(d.getDate(), 2));
		s = s.replace(/%e/g, jsworld._spacePad(d.getDate(), 2));
		s = s.replace(/%F/g, d.getFullYear() +
				         "-" +
					 jsworld._zeroPad(d.getMonth()+1, 2) +
					 "-" +
					 jsworld._zeroPad(d.getDate(), 2));
		s = s.replace(/%h/g, this.lc.abmon[d.getMonth()]); // same as %b
		s = s.replace(/%H/g, jsworld._zeroPad(d.getHours(), 2));
		s = s.replace(/%I/g, jsworld._zeroPad(this._hours12(d.getHours()), 2));
		s = s.replace(/%k/g, d.getHours());
		s = s.replace(/%l/g, this._hours12(d.getHours()));
		s = s.replace(/%m/g, jsworld._zeroPad(d.getMonth()+1, 2));
		s = s.replace(/%n/g, "\n");
		s = s.replace(/%M/g, jsworld._zeroPad(d.getMinutes(), 2));
		s = s.replace(/%p/g, this._getAmPm(d.getHours()));
		s = s.replace(/%P/g, this._getAmPm(d.getHours()).toLocaleLowerCase()); // safe?
		s = s.replace(/%R/g, jsworld._zeroPad(d.getHours(), 2) +
					":" +
					jsworld._zeroPad(d.getMinutes(), 2));
		s = s.replace(/%S/g, jsworld._zeroPad(d.getSeconds(), 2));
		s = s.replace(/%T/g, jsworld._zeroPad(d.getHours(), 2) +
					":" +
					jsworld._zeroPad(d.getMinutes(), 2) +
					":" +
					jsworld._zeroPad(d.getSeconds(), 2));
		s = s.replace(/%w/g, this.lc.day[d.getDay()]);
		s = s.replace(/%y/g, new String(d.getFullYear()).substring(2));
		s = s.replace(/%Y/g, d.getFullYear());
		
		s = s.replace(/%Z/g, ""); // to do: ignored until a reliable TMZ method found
		
		s = s.replace(/%[a-zA-Z]/g, ""); // ignore all other % sequences
		
		return s;
	};
	
	
	/** 
	 * @private 
	 *
	 * @description Does 24 to 12 hour conversion.
	 *
	 * @param {integer Number} hour24 Hour [0..23].
	 * 
	 * @returns {integer Number} Corresponding hour [1..12].
	 */
	this._hours12 = function(hour24) {
		
		if (hour24 === 0)
			return 12; // 00h is 12AM
			
		else if (hour24 > 12)
			return hour24 - 12; // 1PM to 11PM
		
		else
			return hour24; // 1AM to 12PM
	};
	
	
	/** 
	 * @private 
	 * 
	 * @description Gets the appropriate localised AM or PM string depending
	 * on the day hour. Special cases: midnight is 12AM, noon is 12PM.
	 *
	 * @param {integer Number} hour24 Hour [0..23].
	 * 
	 * @returns {String} The corresponding localised AM or PM string.
	 */
	this._getAmPm = function(hour24) {
		
		if (hour24 < 12)
			return this.lc.am;
		else
			return this.lc.pm;
	};
};



/** 
 * @class Class for localised formatting of currency amounts.
 *
 * <p>See: <a href="http://www.opengroup.org/onlinepubs/009695399/basedefs/xbd_chap07.html#tag_07_03_03">
 * POSIX LC_MONETARY</a>.
 *
 * @public
 * @constructor
 * @description Creates a new monetary formatter for the specified locale.
 *
 * @param {jsworld.Locale} locale A locale object specifying the required 
 *        POSIX LC_MONETARY formatting properties.
 * @param {String} [currencyCode] Set the currency explicitly by
 *        passing its international ISO-4217 code, e.g. "USD", "EUR", "GBP".
 *        Use this optional parameter to override the default local currency
 * @param {String} [altIntSymbol] Non-local currencies are formatted
 *        with their international ISO-4217 code to prevent ambiguity.
 *        Use this optional argument to force a different symbol, such as the
 *        currency's shorthand sign. This is mostly useful when the shorthand
 *        sign is both internationally recognised and identifies the currency
 *        uniquely (e.g. the Euro sign).
 *
 * @throws Error on constructor failure.
 */
jsworld.MonetaryFormatter = function(locale, currencyCode, altIntSymbol) {
	
	if (typeof locale != "object" || locale._className != "jsworld.Locale")
		throw "Constructor error: You must provide a valid jsworld.Locale instance";
	
	this.lc = locale;
	
	/** 
	 * @private
	 * @description Lookup table to determine the fraction digits for a
	 * specific currency; most currencies subdivide at 1/100 (2 fractional
	 * digits), so we store only those that deviate from the default.
	 *
	 * <p>The data is from Unicode's CLDR version 1.7.0. The two currencies
	 * with non-decimal subunits (MGA and MRO) are marked as having no
	 * fractional digits as well as all currencies that have no subunits
	 * in circulation.
	 * 
	 * <p>It is "hard-wired" for referential convenience and is only looked
	 * up when an overriding currencyCode parameter is supplied.
	 */
	this.currencyFractionDigits = {
		"AFN" : 0, "ALL" : 0, "AMD" : 0, "BHD" : 3, "BIF" : 0,
		"BYR" : 0, "CLF" : 0, "CLP" : 0, "COP" : 0, "CRC" : 0, 
		"DJF" : 0, "GNF" : 0, "GYD" : 0, "HUF" : 0, "IDR" : 0, 
		"IQD" : 0, "IRR" : 0, "ISK" : 0, "JOD" : 3, "JPY" : 0, 
		"KMF" : 0, "KRW" : 0, "KWD" : 3, "LAK" : 0, "LBP" : 0,
		"LYD" : 3, "MGA" : 0, "MMK" : 0, "MNT" : 0, "MRO" : 0,
		"MUR" : 0, "OMR" : 3, "PKR" : 0, "PYG" : 0, "RSD" : 0, 
		"RWF" : 0, "SLL" : 0, "SOS" : 0, "STD" : 0, "SYP" : 0, 
		"TND" : 3, "TWD" : 0, "TZS" : 0, "UGX" : 0, "UZS" : 0, 
		"VND" : 0, "VUV" : 0, "XAF" : 0, "XOF" : 0, "XPF" : 0, 
		"YER" : 0, "ZMK" : 0
	};
	
	
	// optional currencyCode argument?
	if (typeof currencyCode == "string") {
		// user wanted to override the local currency
		this.currencyCode = currencyCode.toUpperCase();
		
		// must override the frac digits too, for some
		// currencies have 0, 2 or 3!
		var numDigits = this.currencyFractionDigits[this.currencyCode];
		if (typeof numDigits != "number")
			numDigits = 2; // default for most currencies
		this.lc.frac_digits = numDigits;
		this.lc.int_frac_digits = numDigits;
	}
	else {
		// use local currency
		this.currencyCode = this.lc.int_curr_symbol.substring(0,3).toUpperCase();
	}
	
	// extract intl. currency separator
	this.intSep = this.lc.int_curr_symbol.charAt(3);
	
	// flag local or intl. sign formatting?
	if (this.currencyCode == this.lc.int_curr_symbol.substring(0,3)) {
		// currency matches the local one? ->
		// formatting with local symbol and parameters
		this.internationalFormatting = false;
		this.curSym = this.lc.currency_symbol;
	}
	else {
		// currency doesn't match the local ->
		
		// do we have an overriding currency symbol?
		if (typeof altIntSymbol == "string") {
			// -> force formatting with local parameters, using alt symbol
			this.curSym = altIntSymbol;
			this.internationalFormatting = false;
		}
		else {
			// -> force formatting with intl. sign and parameters
			this.internationalFormatting = true;
		}
	}
	
	
	/** 
	 * @public
	 *
	 * @description Gets the currency symbol used in formatting.
	 *
	 * @returns {String} The currency symbol.
	 */
	this.getCurrencySymbol = function() {
		
		return this.curSym;
	};
	
	
	/** 
	 * @public
	 *
	 * @description Gets the position of the currency symbol relative to
	 * the amount. Assumes a non-negative amount and local formatting.
	 *
	 * @param {String} intFlag Optional flag to force international
	 * formatting by passing the string "i".
	 *
	 * @returns {Boolean} True if the symbol precedes the amount, false if
	 * the symbol succeeds the amount.
	 */
	this.currencySymbolPrecedes = function(intFlag) {
		
		if (typeof intFlag == "string" && intFlag == "i") {
			// international formatting was forced
			if (this.lc.int_p_cs_precedes == 1)
				return true;
			else
				return false;
			
		}
		else {
			// check whether local formatting is on or off
			if (this.internationalFormatting) {
				if (this.lc.int_p_cs_precedes == 1)
					return true;
				else
					return false;
			}
			else {
				if (this.lc.p_cs_precedes == 1)
					return true;
				else
					return false;
			}
		}
	};
	
	
	/** 
	 * @public
	 *
	 * @description Gets the decimal delimiter (radix) used in formatting.
	 *
	 * @returns {String} The radix character.
	 */
	this.getDecimalPoint = function() {
		
		return this.lc.mon_decimal_point;
	};
	
	
	/** 
	 * @public
	 *
	 * @description Gets the number of fractional digits. Assumes local
	 * formatting.
	 *
	 * @param {String} intFlag Optional flag to force international
	 *        formatting by passing the string "i".
	 *
	 * @returns {integer Number} The number of fractional digits.
	 */
	this.getFractionalDigits = function(intFlag) {
		
		if (typeof intFlag == "string" && intFlag == "i") {
			// international formatting was forced
			return this.lc.int_frac_digits;
		}
		else {
			// check whether local formatting is on or off
			if (this.internationalFormatting)
				return this.lc.int_frac_digits;
			else
				return this.lc.frac_digits;
		}
	};
	
	
	/** 
	 * @public
	 *
	 * @description Formats a monetary amount according to the preset 
	 * locale.
	 *
	 * <pre>
	 * For local currencies the native shorthand symbol will be used for
	 * formatting.
	 * Example:
	 *        locale is en_US
	 *        currency is USD
	 *        -> the "$" symbol will be used, e.g. $123.45
	 *        
	 * For non-local currencies the international ISO-4217 code will be
	 * used for formatting.
	 * Example:
	 *       locale is en_US (which has USD as currency)
	 *       currency is EUR
	 *       -> the ISO three-letter code will be used, e.g. EUR 123.45
	 *
	 * If the currency is non-local, but an alternative currency symbol was
	 * provided, this will be used instead.
	 * Example
	 *       locale is en_US (which has USD as currency)
	 *       currency is EUR
	 *       an alternative symbol is provided - "€"
	 *       -> the alternative symbol will be used, e.g. €123.45
	 * </pre>
	 * 
	 * @param {Number|String} amount The amount to format as currency.
	 * @param {String} [options] Options to modify the formatted output:
	 *       <ul>
	 *           <li>"^"  suppress grouping
	 *           <li>"!"  suppress the currency symbol
	 *           <li>"~"  suppress the currency symbol and the sign (positive or negative)
	 *           <li>"i"  force international sign (ISO-4217 code) formatting
	 *           <li>".n" specify decimal precision
	 *       
	 * @returns The formatted currency amount as string.
	 *
	 * @throws "Error: Invalid amount" on bad amount.
	 */
	this.format = function(amount, options) {
		
		// if the amount is passed as string, check that it parses to a float
		var floatAmount;
		
		if (typeof amount == "string") {
			amount = jsworld._trim(amount);
			floatAmount = parseFloat(amount);
			
			if (typeof floatAmount != "number" || isNaN(floatAmount))
				throw "Error: Amount string not a number";
		}
		else if (typeof amount == "number") {
			floatAmount = amount;
		}
		else {
			throw "Error: Amount not a number";
		}
		
		// get the required precision, ".n" option arg overrides default locale config
		var reqPrecision = jsworld._getPrecision(options);
		
		if (reqPrecision == -1) {
			if (this.internationalFormatting || jsworld._hasOption("i", options))
				reqPrecision = this.lc.int_frac_digits;
			else
				reqPrecision = this.lc.frac_digits;
		}
		
		// round
		floatAmount = Math.round(floatAmount * Math.pow(10, reqPrecision)) / Math.pow(10, reqPrecision);
		
		
		// convert the float amount to string and parse into
		// object with properties integer and fraction
		var parsedAmount = jsworld._splitNumber(String(floatAmount));
		
		// format integer part with grouping chars
		var formattedIntegerPart;
		
		if (floatAmount === 0)
			formattedIntegerPart = "0";
		else
			formattedIntegerPart = jsworld._hasOption("^", options) ?
				parsedAmount.integer :
				jsworld._formatIntegerPart(parsedAmount.integer, 
				                           this.lc.mon_grouping, 
							   this.lc.mon_thousands_sep);
		
		
		// format the fractional part
		var formattedFractionPart;
		
		if (reqPrecision == -1) {
			// pad fraction with trailing zeros accoring to default locale [int_]frac_digits
			if (this.internationalFormatting || jsworld._hasOption("i", options))
				formattedFractionPart =
					jsworld._formatFractionPart(parsedAmount.fraction, this.lc.int_frac_digits);
			else
				formattedFractionPart =
					jsworld._formatFractionPart(parsedAmount.fraction, this.lc.frac_digits);
		}
		else {
			// pad fraction with trailing zeros according to optional format parameter
			formattedFractionPart =
				jsworld._formatFractionPart(parsedAmount.fraction, reqPrecision);
		}
		
		
		// join integer and decimal parts using the mon_decimal_point property
		var quantity;
		
		if (this.lc.frac_digits > 0 || formattedFractionPart.length)
			quantity = formattedIntegerPart + this.lc.mon_decimal_point + formattedFractionPart;
		else
			quantity = formattedIntegerPart;
		
		
		// do final formatting with sign and symbol
		if (jsworld._hasOption("~", options)) {
			return quantity;
		}
		else {
			var suppressSymbol = jsworld._hasOption("!", options) ? true : false;
			
			var sign = floatAmount < 0 ? "-" : "+";
			
			if (this.internationalFormatting || jsworld._hasOption("i", options)) {
				
				// format with ISO-4217 code (suppressed or not)
				if (suppressSymbol)
					return this._formatAsInternationalCurrencyWithNoSym(sign, quantity);
				else
					return this._formatAsInternationalCurrency(sign, quantity);
			}
			else {
				// format with local currency code (suppressed or not)
				if (suppressSymbol)
					return this._formatAsLocalCurrencyWithNoSym(sign, quantity);
				else
					return this._formatAsLocalCurrency(sign, quantity);
			}
		}
	};
	
	
	/** 
	 * @private
	 *
	 * @description Assembles the final string with sign, separator and symbol as local
	 * currency.
	 *
	 * @param {String} sign The amount sign: "+" or "-".
	 * @param {String} q The formatted quantity (unsigned).
	 *
	 * @returns {String} The final formatted string.
	 */
	this._formatAsLocalCurrency = function (sign, q) {
		
		// assemble final formatted amount by going over all possible value combinations of:
		// sign {+,-} , sign position {0,1,2,3,4} , separator {0,1,2} , symbol position {0,1}
		if (sign == "+") {
			
			// parentheses
			if      (this.lc.p_sign_posn === 0 && this.lc.p_sep_by_space === 0 && this.lc.p_cs_precedes === 0) {
				return "(" + q + this.curSym + ")";
			}
			else if (this.lc.p_sign_posn === 0 && this.lc.p_sep_by_space === 0 && this.lc.p_cs_precedes === 1) {
				return "(" + this.curSym + q + ")";
			}
			else if (this.lc.p_sign_posn === 0 && this.lc.p_sep_by_space === 1 && this.lc.p_cs_precedes === 0) {
				return "(" + q + " " + this.curSym + ")";
			}
			else if (this.lc.p_sign_posn === 0 && this.lc.p_sep_by_space === 1 && this.lc.p_cs_precedes === 1) {
				return "(" + this.curSym + " " + q + ")";
			}
			
			// sign before q + sym
			else if (this.lc.p_sign_posn === 1 && this.lc.p_sep_by_space === 0 && this.lc.p_cs_precedes === 0) {
				return this.lc.positive_sign + q + this.curSym;
			}
			else if (this.lc.p_sign_posn === 1 && this.lc.p_sep_by_space === 0 && this.lc.p_cs_precedes === 1) {
				return this.lc.positive_sign + this.curSym + q;
			}
			else if (this.lc.p_sign_posn === 1 && this.lc.p_sep_by_space === 1 && this.lc.p_cs_precedes === 0) {
				return this.lc.positive_sign + q + " " + this.curSym;
			}
			else if (this.lc.p_sign_posn === 1 && this.lc.p_sep_by_space === 1 && this.lc.p_cs_precedes === 1) {
				return this.lc.positive_sign + this.curSym + " " + q;
			}
			else if (this.lc.p_sign_posn === 1 && this.lc.p_sep_by_space === 2 && this.lc.p_cs_precedes === 0) {
				return this.lc.positive_sign + " " + q + this.curSym;
			}
			else if (this.lc.p_sign_posn === 1 && this.lc.p_sep_by_space === 2 && this.lc.p_cs_precedes === 1) {
				return this.lc.positive_sign + " " + this.curSym + q;
			}
			
			// sign after q + sym
			else if (this.lc.p_sign_posn === 2 && this.lc.p_sep_by_space === 0 && this.lc.p_cs_precedes === 0) {
				return q + this.curSym + this.lc.positive_sign;
			}
			else if (this.lc.p_sign_posn === 2 && this.lc.p_sep_by_space === 0 && this.lc.p_cs_precedes === 1) {
				return this.curSym + q + this.lc.positive_sign;
			}
			else if (this.lc.p_sign_posn === 2 && this.lc.p_sep_by_space === 1 && this.lc.p_cs_precedes === 0) {
				return  q + " " + this.curSym + this.lc.positive_sign;
			}
			else if (this.lc.p_sign_posn === 2 && this.lc.p_sep_by_space === 1 && this.lc.p_cs_precedes === 1) {
				return this.curSym + " " + q + this.lc.positive_sign;
			}
			else if (this.lc.p_sign_posn === 2 && this.lc.p_sep_by_space === 2 && this.lc.p_cs_precedes === 0) {
				return q + this.curSym + " " + this.lc.positive_sign;
			}
			else if (this.lc.p_sign_posn === 2 && this.lc.p_sep_by_space === 2 && this.lc.p_cs_precedes === 1) {
				return this.curSym + q + " " + this.lc.positive_sign;
			}
			
			// sign before sym
			else if (this.lc.p_sign_posn === 3 && this.lc.p_sep_by_space === 0 && this.lc.p_cs_precedes === 0) {
				return q + this.lc.positive_sign + this.curSym;
			}
			else if (this.lc.p_sign_posn === 3 && this.lc.p_sep_by_space === 0 && this.lc.p_cs_precedes === 1) {
				return this.lc.positive_sign + this.curSym + q;
			}
			else if (this.lc.p_sign_posn === 3 && this.lc.p_sep_by_space === 1 && this.lc.p_cs_precedes === 0) {
				return q + " " + this.lc.positive_sign + this.curSym;
			}
			else if (this.lc.p_sign_posn === 3 && this.lc.p_sep_by_space === 1 && this.lc.p_cs_precedes === 1) {
				return this.lc.positive_sign + this.curSym + " " + q;
			}
			else if (this.lc.p_sign_posn === 3 && this.lc.p_sep_by_space === 2 && this.lc.p_cs_precedes === 0) {
				return q + this.lc.positive_sign + " " + this.curSym;
			}
			else if (this.lc.p_sign_posn === 3 && this.lc.p_sep_by_space === 2 && this.lc.p_cs_precedes === 1) {
				return this.lc.positive_sign + " " + this.curSym + q;
			}
			
			// sign after symbol
			else if (this.lc.p_sign_posn === 4 && this.lc.p_sep_by_space === 0 && this.lc.p_cs_precedes === 0) {
				return q + this.curSym + this.lc.positive_sign;
			}
			else if (this.lc.p_sign_posn === 4 && this.lc.p_sep_by_space === 0 && this.lc.p_cs_precedes === 1) {
				return this.curSym + this.lc.positive_sign + q;
			}
			else if (this.lc.p_sign_posn === 4 && this.lc.p_sep_by_space === 1 && this.lc.p_cs_precedes === 0) {
				return  q + " " + this.curSym + this.lc.positive_sign;
			}
			else if (this.lc.p_sign_posn === 4 && this.lc.p_sep_by_space === 1 && this.lc.p_cs_precedes === 1) {
				return this.curSym + this.lc.positive_sign + " " + q;
			}
			else if (this.lc.p_sign_posn === 4 && this.lc.p_sep_by_space === 2 && this.lc.p_cs_precedes === 0) {
				return q + this.curSym + " " + this.lc.positive_sign;
			}
			else if (this.lc.p_sign_posn === 4 && this.lc.p_sep_by_space === 2 && this.lc.p_cs_precedes === 1) {
				return this.curSym + " " + this.lc.positive_sign + q;
			}
			
		}
		else if (sign == "-") {
			
			// parentheses enclose q + sym
			if      (this.lc.n_sign_posn === 0 && this.lc.n_sep_by_space === 0 && this.lc.n_cs_precedes === 0) {
				return "(" + q + this.curSym + ")";
			}
			else if (this.lc.n_sign_posn === 0 && this.lc.n_sep_by_space === 0 && this.lc.n_cs_precedes === 1) {
				return "(" + this.curSym + q + ")";
			}
			else if (this.lc.n_sign_posn === 0 && this.lc.n_sep_by_space === 1 && this.lc.n_cs_precedes === 0) {
				return "(" + q + " " + this.curSym + ")";
			}
			else if (this.lc.n_sign_posn === 0 && this.lc.n_sep_by_space === 1 && this.lc.n_cs_precedes === 1) {
				return "(" + this.curSym + " " + q + ")";
			}
			
			// sign before q + sym
			else if (this.lc.n_sign_posn === 1 && this.lc.n_sep_by_space === 0 && this.lc.n_cs_precedes === 0) {
				return this.lc.negative_sign + q + this.curSym;
			}
			else if (this.lc.n_sign_posn === 1 && this.lc.n_sep_by_space === 0 && this.lc.n_cs_precedes === 1) {
				return this.lc.negative_sign + this.curSym + q;
			}
			else if (this.lc.n_sign_posn === 1 && this.lc.n_sep_by_space === 1 && this.lc.n_cs_precedes === 0) {
				return this.lc.negative_sign + q + " " + this.curSym;
			}
			else if (this.lc.n_sign_posn === 1 && this.lc.n_sep_by_space === 1 && this.lc.n_cs_precedes === 1) {
				return this.lc.negative_sign + this.curSym + " " + q;
			}
			else if (this.lc.n_sign_posn === 1 && this.lc.n_sep_by_space === 2 && this.lc.n_cs_precedes === 0) {
				return this.lc.negative_sign + " " + q + this.curSym;
			}
			else if (this.lc.n_sign_posn === 1 && this.lc.n_sep_by_space === 2 && this.lc.n_cs_precedes === 1) {
				return this.lc.negative_sign + " " + this.curSym + q;
			}
			
			// sign after q + sym
			else if (this.lc.n_sign_posn === 2 && this.lc.n_sep_by_space === 0 && this.lc.n_cs_precedes === 0) {
				return q + this.curSym + this.lc.negative_sign;
			}
			else if (this.lc.n_sign_posn === 2 && this.lc.n_sep_by_space === 0 && this.lc.n_cs_precedes === 1) {
				return this.curSym + q + this.lc.negative_sign;
			}
			else if (this.lc.n_sign_posn === 2 && this.lc.n_sep_by_space === 1 && this.lc.n_cs_precedes === 0) {
				return  q + " " + this.curSym + this.lc.negative_sign;
			}
			else if (this.lc.n_sign_posn === 2 && this.lc.n_sep_by_space === 1 && this.lc.n_cs_precedes === 1) {
				return this.curSym + " " + q + this.lc.negative_sign;
			}
			else if (this.lc.n_sign_posn === 2 && this.lc.n_sep_by_space === 2 && this.lc.n_cs_precedes === 0) {
				return q + this.curSym + " " + this.lc.negative_sign;
			}
			else if (this.lc.n_sign_posn === 2 && this.lc.n_sep_by_space === 2 && this.lc.n_cs_precedes === 1) {
				return this.curSym + q + " " + this.lc.negative_sign;
			}
			
			// sign before sym
			else if (this.lc.n_sign_posn === 3 && this.lc.n_sep_by_space === 0 && this.lc.n_cs_precedes === 0) {
				return q + this.lc.negative_sign + this.curSym;
			}
			else if (this.lc.n_sign_posn === 3 && this.lc.n_sep_by_space === 0 && this.lc.n_cs_precedes === 1) {
				return this.lc.negative_sign + this.curSym + q;
			}
			else if (this.lc.n_sign_posn === 3 && this.lc.n_sep_by_space === 1 && this.lc.n_cs_precedes === 0) {
				return q + " " + this.lc.negative_sign + this.curSym;
			}
			else if (this.lc.n_sign_posn === 3 && this.lc.n_sep_by_space === 1 && this.lc.n_cs_precedes === 1) {
				return this.lc.negative_sign + this.curSym + " " + q;
			}
			else if (this.lc.n_sign_posn === 3 && this.lc.n_sep_by_space === 2 && this.lc.n_cs_precedes === 0) {
				return q + this.lc.negative_sign + " " + this.curSym;
			}
			else if (this.lc.n_sign_posn === 3 && this.lc.n_sep_by_space === 2 && this.lc.n_cs_precedes === 1) {
				return this.lc.negative_sign + " " + this.curSym + q;
			}
			
			// sign after symbol
			else if (this.lc.n_sign_posn === 4 && this.lc.n_sep_by_space === 0 && this.lc.n_cs_precedes === 0) {
				return q + this.curSym + this.lc.negative_sign;
			}
			else if (this.lc.n_sign_posn === 4 && this.lc.n_sep_by_space === 0 && this.lc.n_cs_precedes === 1) {
				return this.curSym + this.lc.negative_sign + q;
			}
			else if (this.lc.n_sign_posn === 4 && this.lc.n_sep_by_space === 1 && this.lc.n_cs_precedes === 0) {
				return  q + " " + this.curSym + this.lc.negative_sign;
			}
			else if (this.lc.n_sign_posn === 4 && this.lc.n_sep_by_space === 1 && this.lc.n_cs_precedes === 1) {
				return this.curSym + this.lc.negative_sign + " " + q;
			}
			else if (this.lc.n_sign_posn === 4 && this.lc.n_sep_by_space === 2 && this.lc.n_cs_precedes === 0) {
				return q + this.curSym + " " + this.lc.negative_sign;
			}
			else if (this.lc.n_sign_posn === 4 && this.lc.n_sep_by_space === 2 && this.lc.n_cs_precedes === 1) {
				return this.curSym + " " + this.lc.negative_sign + q;
			}
		}
		
		// throw error if we fall through
		throw "Error: Invalid POSIX LC MONETARY definition";
	};
	
	
	/** 
	 * @private
	 *
	 * @description Assembles the final string with sign, separator and ISO-4217
	 * currency code.
	 *
	 * @param {String} sign The amount sign: "+" or "-".
	 * @param {String} q The formatted quantity (unsigned).
	 *
	 * @returns {String} The final formatted string.
	 */
	this._formatAsInternationalCurrency = function (sign, q) {
		
		// assemble the final formatted amount by going over all possible value combinations of:
		// sign {+,-} , sign position {0,1,2,3,4} , separator {0,1,2} , symbol position {0,1}
		
		if (sign == "+") {
			
			// parentheses
			if      (this.lc.int_p_sign_posn === 0 && this.lc.int_p_sep_by_space === 0 && this.lc.int_p_cs_precedes === 0) {
				return "(" + q + this.currencyCode + ")";
			}
			else if (this.lc.int_p_sign_posn === 0 && this.lc.int_p_sep_by_space === 0 && this.lc.int_p_cs_precedes === 1) {
				return "(" + this.currencyCode + q + ")";
			}
			else if (this.lc.int_p_sign_posn === 0 && this.lc.int_p_sep_by_space === 1 && this.lc.int_p_cs_precedes === 0) {
				return "(" + q + this.intSep + this.currencyCode + ")";
			}
			else if (this.lc.int_p_sign_posn === 0 && this.lc.int_p_sep_by_space === 1 && this.lc.int_p_cs_precedes === 1) {
				return "(" + this.currencyCode + this.intSep + q + ")";
			}
			
			// sign before q + sym
			else if (this.lc.int_p_sign_posn === 1 && this.lc.int_p_sep_by_space === 0 && this.lc.int_p_cs_precedes === 0) {
				return this.lc.positive_sign + q + this.currencyCode;
			}
			else if (this.lc.int_p_sign_posn === 1 && this.lc.int_p_sep_by_space === 0 && this.lc.int_p_cs_precedes === 1) {
				return this.lc.positive_sign + this.currencyCode + q;
			}
			else if (this.lc.int_p_sign_posn === 1 && this.lc.int_p_sep_by_space === 1 && this.lc.int_p_cs_precedes === 0) {
				return this.lc.positive_sign + q + this.intSep + this.currencyCode;
			}
			else if (this.lc.int_p_sign_posn === 1 && this.lc.int_p_sep_by_space === 1 && this.lc.int_p_cs_precedes === 1) {
				return this.lc.positive_sign + this.currencyCode + this.intSep + q;
			}
			else if (this.lc.int_p_sign_posn === 1 && this.lc.int_p_sep_by_space === 2 && this.lc.int_p_cs_precedes === 0) {
				return this.lc.positive_sign + this.intSep + q + this.currencyCode;
			}
			else if (this.lc.int_p_sign_posn === 1 && this.lc.int_p_sep_by_space === 2 && this.lc.int_p_cs_precedes === 1) {
				return this.lc.positive_sign + this.intSep + this.currencyCode + q;
			}
			
			// sign after q + sym
			else if (this.lc.int_p_sign_posn === 2 && this.lc.int_p_sep_by_space === 0 && this.lc.int_p_cs_precedes === 0) {
				return q + this.currencyCode + this.lc.positive_sign;
			}
			else if (this.lc.int_p_sign_posn === 2 && this.lc.int_p_sep_by_space === 0 && this.lc.int_p_cs_precedes === 1) {
				return this.currencyCode + q + this.lc.positive_sign;
			}
			else if (this.lc.int_p_sign_posn === 2 && this.lc.int_p_sep_by_space === 1 && this.lc.int_p_cs_precedes === 0) {
				return  q + this.intSep + this.currencyCode + this.lc.positive_sign;
			}
			else if (this.lc.int_p_sign_posn === 2 && this.lc.int_p_sep_by_space === 1 && this.lc.int_p_cs_precedes === 1) {
				return this.currencyCode + this.intSep + q + this.lc.positive_sign;
			}
			else if (this.lc.int_p_sign_posn === 2 && this.lc.int_p_sep_by_space === 2 && this.lc.int_p_cs_precedes === 0) {
				return q + this.currencyCode + this.intSep + this.lc.positive_sign;
			}
			else if (this.lc.int_p_sign_posn === 2 && this.lc.int_p_sep_by_space === 2 && this.lc.int_p_cs_precedes === 1) {
				return this.currencyCode + q + this.intSep + this.lc.positive_sign;
			}
			
			// sign before sym
			else if (this.lc.int_p_sign_posn === 3 && this.lc.int_p_sep_by_space === 0 && this.lc.int_p_cs_precedes === 0) {
				return q + this.lc.positive_sign + this.currencyCode;
			}
			else if (this.lc.int_p_sign_posn === 3 && this.lc.int_p_sep_by_space === 0 && this.lc.int_p_cs_precedes === 1) {
				return this.lc.positive_sign + this.currencyCode + q;
			}
			else if (this.lc.int_p_sign_posn === 3 && this.lc.int_p_sep_by_space === 1 && this.lc.int_p_cs_precedes === 0) {
				return q + this.intSep + this.lc.positive_sign + this.currencyCode;
			}
			else if (this.lc.int_p_sign_posn === 3 && this.lc.int_p_sep_by_space === 1 && this.lc.int_p_cs_precedes === 1) {
				return this.lc.positive_sign + this.currencyCode + this.intSep + q;
			}
			else if (this.lc.int_p_sign_posn === 3 && this.lc.int_p_sep_by_space === 2 && this.lc.int_p_cs_precedes === 0) {
				return q + this.lc.positive_sign + this.intSep + this.currencyCode;
			}
			else if (this.lc.int_p_sign_posn === 3 && this.lc.int_p_sep_by_space === 2 && this.lc.int_p_cs_precedes === 1) {
				return this.lc.positive_sign + this.intSep + this.currencyCode + q;
			}
			
			// sign after symbol
			else if (this.lc.int_p_sign_posn === 4 && this.lc.int_p_sep_by_space === 0 && this.lc.int_p_cs_precedes === 0) {
				return q + this.currencyCode + this.lc.positive_sign;
			}
			else if (this.lc.int_p_sign_posn === 4 && this.lc.int_p_sep_by_space === 0 && this.lc.int_p_cs_precedes === 1) {
				return this.currencyCode + this.lc.positive_sign + q;
			}
			else if (this.lc.int_p_sign_posn === 4 && this.lc.int_p_sep_by_space === 1 && this.lc.int_p_cs_precedes === 0) {
				return  q + this.intSep + this.currencyCode + this.lc.positive_sign;
			}
			else if (this.lc.int_p_sign_posn === 4 && this.lc.int_p_sep_by_space === 1 && this.lc.int_p_cs_precedes === 1) {
				return this.currencyCode + this.lc.positive_sign + this.intSep + q;
			}
			else if (this.lc.int_p_sign_posn === 4 && this.lc.int_p_sep_by_space === 2 && this.lc.int_p_cs_precedes === 0) {
				return q + this.currencyCode + this.intSep + this.lc.positive_sign;
			}
			else if (this.lc.int_p_sign_posn === 4 && this.lc.int_p_sep_by_space === 2 && this.lc.int_p_cs_precedes === 1) {
				return this.currencyCode + this.intSep + this.lc.positive_sign + q;
			}
			
		}
		else if (sign == "-") {
			
			// parentheses enclose q + sym
			if      (this.lc.int_n_sign_posn === 0 && this.lc.int_n_sep_by_space === 0 && this.lc.int_n_cs_precedes === 0) {
				return "(" + q + this.currencyCode + ")";
			}
			else if (this.lc.int_n_sign_posn === 0 && this.lc.int_n_sep_by_space === 0 && this.lc.int_n_cs_precedes === 1) {
				return "(" + this.currencyCode + q + ")";
			}
			else if (this.lc.int_n_sign_posn === 0 && this.lc.int_n_sep_by_space === 1 && this.lc.int_n_cs_precedes === 0) {
				return "(" + q + this.intSep + this.currencyCode + ")";
			}
			else if (this.lc.int_n_sign_posn === 0 && this.lc.int_n_sep_by_space === 1 && this.lc.int_n_cs_precedes === 1) {
				return "(" + this.currencyCode + this.intSep + q + ")";
			}
			
			// sign before q + sym
			else if (this.lc.int_n_sign_posn === 1 && this.lc.int_n_sep_by_space === 0 && this.lc.int_n_cs_precedes === 0) {
				return this.lc.negative_sign + q + this.currencyCode;
			}
			else if (this.lc.int_n_sign_posn === 1 && this.lc.int_n_sep_by_space === 0 && this.lc.int_n_cs_precedes === 1) {
				return this.lc.negative_sign + this.currencyCode + q;
			}
			else if (this.lc.int_n_sign_posn === 1 && this.lc.int_n_sep_by_space === 1 && this.lc.int_n_cs_precedes === 0) {
				return this.lc.negative_sign + q + this.intSep + this.currencyCode;
			}
			else if (this.lc.int_n_sign_posn === 1 && this.lc.int_n_sep_by_space === 1 && this.lc.int_n_cs_precedes === 1) {
				return this.lc.negative_sign + this.currencyCode + this.intSep + q;
			}
			else if (this.lc.int_n_sign_posn === 1 && this.lc.int_n_sep_by_space === 2 && this.lc.int_n_cs_precedes === 0) {
				return this.lc.negative_sign + this.intSep + q + this.currencyCode;
			}
			else if (this.lc.int_n_sign_posn === 1 && this.lc.int_n_sep_by_space === 2 && this.lc.int_n_cs_precedes === 1) {
				return this.lc.negative_sign + this.intSep + this.currencyCode + q;
			}
			
			// sign after q + sym
			else if (this.lc.int_n_sign_posn === 2 && this.lc.int_n_sep_by_space === 0 && this.lc.int_n_cs_precedes === 0) {
				return q + this.currencyCode + this.lc.negative_sign;
			}
			else if (this.lc.int_n_sign_posn === 2 && this.lc.int_n_sep_by_space === 0 && this.lc.int_n_cs_precedes === 1) {
				return this.currencyCode + q + this.lc.negative_sign;
			}
			else if (this.lc.int_n_sign_posn === 2 && this.lc.int_n_sep_by_space === 1 && this.lc.int_n_cs_precedes === 0) {
				return  q + this.intSep + this.currencyCode + this.lc.negative_sign;
			}
			else if (this.lc.int_n_sign_posn === 2 && this.lc.int_n_sep_by_space === 1 && this.lc.int_n_cs_precedes === 1) {
				return this.currencyCode + this.intSep + q + this.lc.negative_sign;
			}
			else if (this.lc.int_n_sign_posn === 2 && this.lc.int_n_sep_by_space === 2 && this.lc.int_n_cs_precedes === 0) {
				return q + this.currencyCode + this.intSep + this.lc.negative_sign;
			}
			else if (this.lc.int_n_sign_posn === 2 && this.lc.int_n_sep_by_space === 2 && this.lc.int_n_cs_precedes === 1) {
				return this.currencyCode + q + this.intSep + this.lc.negative_sign;
			}
			
			// sign before sym
			else if (this.lc.int_n_sign_posn === 3 && this.lc.int_n_sep_by_space === 0 && this.lc.int_n_cs_precedes === 0) {
				return q + this.lc.negative_sign + this.currencyCode;
			}
			else if (this.lc.int_n_sign_posn === 3 && this.lc.int_n_sep_by_space === 0 && this.lc.int_n_cs_precedes === 1) {
				return this.lc.negative_sign + this.currencyCode + q;
			}
			else if (this.lc.int_n_sign_posn === 3 && this.lc.int_n_sep_by_space === 1 && this.lc.int_n_cs_precedes === 0) {
				return q + this.intSep + this.lc.negative_sign + this.currencyCode;
			}
			else if (this.lc.int_n_sign_posn === 3 && this.lc.int_n_sep_by_space === 1 && this.lc.int_n_cs_precedes === 1) {
				return this.lc.negative_sign + this.currencyCode + this.intSep + q;
			}
			else if (this.lc.int_n_sign_posn === 3 && this.lc.int_n_sep_by_space === 2 && this.lc.int_n_cs_precedes === 0) {
				return q + this.lc.negative_sign + this.intSep + this.currencyCode;
			}
			else if (this.lc.int_n_sign_posn === 3 && this.lc.int_n_sep_by_space === 2 && this.lc.int_n_cs_precedes === 1) {
				return this.lc.negative_sign + this.intSep + this.currencyCode + q;
			}
			
			// sign after symbol
			else if (this.lc.int_n_sign_posn === 4 && this.lc.int_n_sep_by_space === 0 && this.lc.int_n_cs_precedes === 0) {
				return q + this.currencyCode + this.lc.negative_sign;
			}
			else if (this.lc.int_n_sign_posn === 4 && this.lc.int_n_sep_by_space === 0 && this.lc.int_n_cs_precedes === 1) {
				return this.currencyCode + this.lc.negative_sign + q;
			}
			else if (this.lc.int_n_sign_posn === 4 && this.lc.int_n_sep_by_space === 1 && this.lc.int_n_cs_precedes === 0) {
				return  q + this.intSep + this.currencyCode + this.lc.negative_sign;
			}
			else if (this.lc.int_n_sign_posn === 4 && this.lc.int_n_sep_by_space === 1 && this.lc.int_n_cs_precedes === 1) {
				return this.currencyCode + this.lc.negative_sign + this.intSep + q;
			}
			else if (this.lc.int_n_sign_posn === 4 && this.lc.int_n_sep_by_space === 2 && this.lc.int_n_cs_precedes === 0) {
				return q + this.currencyCode + this.intSep + this.lc.negative_sign;
			}
			else if (this.lc.int_n_sign_posn === 4 && this.lc.int_n_sep_by_space === 2 && this.lc.int_n_cs_precedes === 1) {
				return this.currencyCode + this.intSep + this.lc.negative_sign + q;
			}
		}
		
		// throw error if we fall through
		throw "Error: Invalid POSIX LC MONETARY definition";
	};
	
	
	/** 
	 * @private
	 *
	 * @description Assembles the final string with sign and separator, but suppress the
	 * local currency symbol.
	 *
	 * @param {String} sign The amount sign: "+" or "-".
	 * @param {String} q The formatted quantity (unsigned).
	 *
	 * @returns {String} The final formatted string
	 */
	this._formatAsLocalCurrencyWithNoSym = function (sign, q) {
		
		// assemble the final formatted amount by going over all possible value combinations of:
		// sign {+,-} , sign position {0,1,2,3,4} , separator {0,1,2} , symbol position {0,1}
		
		if (sign == "+") {
			
			// parentheses
			if      (this.lc.p_sign_posn === 0) {
				return "(" + q + ")";
			}
			
			// sign before q + sym
			else if (this.lc.p_sign_posn === 1 && this.lc.p_sep_by_space === 0 && this.lc.p_cs_precedes === 0) {
				return this.lc.positive_sign + q;
			}
			else if (this.lc.p_sign_posn === 1 && this.lc.p_sep_by_space === 0 && this.lc.p_cs_precedes === 1) {
				return this.lc.positive_sign + q;
			}
			else if (this.lc.p_sign_posn === 1 && this.lc.p_sep_by_space === 1 && this.lc.p_cs_precedes === 0) {
				return this.lc.positive_sign + q;
			}
			else if (this.lc.p_sign_posn === 1 && this.lc.p_sep_by_space === 1 && this.lc.p_cs_precedes === 1) {
				return this.lc.positive_sign + q;
			}
			else if (this.lc.p_sign_posn === 1 && this.lc.p_sep_by_space === 2 && this.lc.p_cs_precedes === 0) {
				return this.lc.positive_sign + " " + q;
			}
			else if (this.lc.p_sign_posn === 1 && this.lc.p_sep_by_space === 2 && this.lc.p_cs_precedes === 1) {
				return this.lc.positive_sign + " " + q;
			}
			
			// sign after q + sym
			else if (this.lc.p_sign_posn === 2 && this.lc.p_sep_by_space === 0 && this.lc.p_cs_precedes === 0) {
				return q + this.lc.positive_sign;
			}
			else if (this.lc.p_sign_posn === 2 && this.lc.p_sep_by_space === 0 && this.lc.p_cs_precedes === 1) {
				return q + this.lc.positive_sign;
			}
			else if (this.lc.p_sign_posn === 2 && this.lc.p_sep_by_space === 1 && this.lc.p_cs_precedes === 0) {
				return  q + " " + this.lc.positive_sign;
			}
			else if (this.lc.p_sign_posn === 2 && this.lc.p_sep_by_space === 1 && this.lc.p_cs_precedes === 1) {
				return q + this.lc.positive_sign;
			}
			else if (this.lc.p_sign_posn === 2 && this.lc.p_sep_by_space === 2 && this.lc.p_cs_precedes === 0) {
				return q + this.lc.positive_sign;
			}
			else if (this.lc.p_sign_posn === 2 && this.lc.p_sep_by_space === 2 && this.lc.p_cs_precedes === 1) {
				return q + " " + this.lc.positive_sign;
			}
			
			// sign before sym
			else if (this.lc.p_sign_posn === 3 && this.lc.p_sep_by_space === 0 && this.lc.p_cs_precedes === 0) {
				return q + this.lc.positive_sign;
			}
			else if (this.lc.p_sign_posn === 3 && this.lc.p_sep_by_space === 0 && this.lc.p_cs_precedes === 1) {
				return this.lc.positive_sign + q;
			}
			else if (this.lc.p_sign_posn === 3 && this.lc.p_sep_by_space === 1 && this.lc.p_cs_precedes === 0) {
				return q + " " + this.lc.positive_sign;
			}
			else if (this.lc.p_sign_posn === 3 && this.lc.p_sep_by_space === 1 && this.lc.p_cs_precedes === 1) {
				return this.lc.positive_sign + " " + q;
			}
			else if (this.lc.p_sign_posn === 3 && this.lc.p_sep_by_space === 2 && this.lc.p_cs_precedes === 0) {
				return q + this.lc.positive_sign;
			}
			else if (this.lc.p_sign_posn === 3 && this.lc.p_sep_by_space === 2 && this.lc.p_cs_precedes === 1) {
				return this.lc.positive_sign + " " + q;
			}
			
			// sign after symbol
			else if (this.lc.p_sign_posn === 4 && this.lc.p_sep_by_space === 0 && this.lc.p_cs_precedes === 0) {
				return q + this.lc.positive_sign;
			}
			else if (this.lc.p_sign_posn === 4 && this.lc.p_sep_by_space === 0 && this.lc.p_cs_precedes === 1) {
				return this.lc.positive_sign + q;
			}
			else if (this.lc.p_sign_posn === 4 && this.lc.p_sep_by_space === 1 && this.lc.p_cs_precedes === 0) {
				return  q + " " + this.lc.positive_sign;
			}
			else if (this.lc.p_sign_posn === 4 && this.lc.p_sep_by_space === 1 && this.lc.p_cs_precedes === 1) {
				return this.lc.positive_sign + " " + q;
			}
			else if (this.lc.p_sign_posn === 4 && this.lc.p_sep_by_space === 2 && this.lc.p_cs_precedes === 0) {
				return q + " " + this.lc.positive_sign;
			}
			else if (this.lc.p_sign_posn === 4 && this.lc.p_sep_by_space === 2 && this.lc.p_cs_precedes === 1) {
				return this.lc.positive_sign + q;
			}
			
		}
		else if (sign == "-") {
			
			// parentheses enclose q + sym
			if      (this.lc.n_sign_posn === 0) {
				return "(" + q + ")";
			}
			
			// sign before q + sym
			else if (this.lc.n_sign_posn === 1 && this.lc.n_sep_by_space === 0 && this.lc.n_cs_precedes === 0) {
				return this.lc.negative_sign + q;
			}
			else if (this.lc.n_sign_posn === 1 && this.lc.n_sep_by_space === 0 && this.lc.n_cs_precedes === 1) {
				return this.lc.negative_sign + q;
			}
			else if (this.lc.n_sign_posn === 1 && this.lc.n_sep_by_space === 1 && this.lc.n_cs_precedes === 0) {
				return this.lc.negative_sign + q;
			}
			else if (this.lc.n_sign_posn === 1 && this.lc.n_sep_by_space === 1 && this.lc.n_cs_precedes === 1) {
				return this.lc.negative_sign + " " + q;
			}
			else if (this.lc.n_sign_posn === 1 && this.lc.n_sep_by_space === 2 && this.lc.n_cs_precedes === 0) {
				return this.lc.negative_sign + " " + q;
			}
			else if (this.lc.n_sign_posn === 1 && this.lc.n_sep_by_space === 2 && this.lc.n_cs_precedes === 1) {
				return this.lc.negative_sign + " " + q;
			}
			
			// sign after q + sym
			else if (this.lc.n_sign_posn === 2 && this.lc.n_sep_by_space === 0 && this.lc.n_cs_precedes === 0) {
				return q + this.lc.negative_sign;
			}
			else if (this.lc.n_sign_posn === 2 && this.lc.n_sep_by_space === 0 && this.lc.n_cs_precedes === 1) {
				return q + this.lc.negative_sign;
			}
			else if (this.lc.n_sign_posn === 2 && this.lc.n_sep_by_space === 1 && this.lc.n_cs_precedes === 0) {
				return  q + " " + this.lc.negative_sign;
			}
			else if (this.lc.n_sign_posn === 2 && this.lc.n_sep_by_space === 1 && this.lc.n_cs_precedes === 1) {
				return q + this.lc.negative_sign;
			}
			else if (this.lc.n_sign_posn === 2 && this.lc.n_sep_by_space === 2 && this.lc.n_cs_precedes === 0) {
				return q + " " + this.lc.negative_sign;
			}
			else if (this.lc.n_sign_posn === 2 && this.lc.n_sep_by_space === 2 && this.lc.n_cs_precedes === 1) {
				return q + " " + this.lc.negative_sign;
			}
			
			// sign before sym
			else if (this.lc.n_sign_posn === 3 && this.lc.n_sep_by_space === 0 && this.lc.n_cs_precedes === 0) {
				return q + this.lc.negative_sign;
			}
			else if (this.lc.n_sign_posn === 3 && this.lc.n_sep_by_space === 0 && this.lc.n_cs_precedes === 1) {
				return this.lc.negative_sign + q;
			}
			else if (this.lc.n_sign_posn === 3 && this.lc.n_sep_by_space === 1 && this.lc.n_cs_precedes === 0) {
				return q + " " + this.lc.negative_sign;
			}
			else if (this.lc.n_sign_posn === 3 && this.lc.n_sep_by_space === 1 && this.lc.n_cs_precedes === 1) {
				return this.lc.negative_sign + " " + q;
			}
			else if (this.lc.n_sign_posn === 3 && this.lc.n_sep_by_space === 2 && this.lc.n_cs_precedes === 0) {
				return q + this.lc.negative_sign;
			}
			else if (this.lc.n_sign_posn === 3 && this.lc.n_sep_by_space === 2 && this.lc.n_cs_precedes === 1) {
				return this.lc.negative_sign + " " + q;
			}
			
			// sign after symbol
			else if (this.lc.n_sign_posn === 4 && this.lc.n_sep_by_space === 0 && this.lc.n_cs_precedes === 0) {
				return q + this.lc.negative_sign;
			}
			else if (this.lc.n_sign_posn === 4 && this.lc.n_sep_by_space === 0 && this.lc.n_cs_precedes === 1) {
				return this.lc.negative_sign + q;
			}
			else if (this.lc.n_sign_posn === 4 && this.lc.n_sep_by_space === 1 && this.lc.n_cs_precedes === 0) {
				return  q + " " + this.lc.negative_sign;
			}
			else if (this.lc.n_sign_posn === 4 && this.lc.n_sep_by_space === 1 && this.lc.n_cs_precedes === 1) {
				return this.lc.negative_sign + " " + q;
			}
			else if (this.lc.n_sign_posn === 4 && this.lc.n_sep_by_space === 2 && this.lc.n_cs_precedes === 0) {
				return q + " " + this.lc.negative_sign;
			}
			else if (this.lc.n_sign_posn === 4 && this.lc.n_sep_by_space === 2 && this.lc.n_cs_precedes === 1) {
				return this.lc.negative_sign + q;
			}
		}
		
		// throw error if we fall through
		throw "Error: Invalid POSIX LC MONETARY definition";
	};
	
	
	/** 
	 * @private
	 *
	 * @description Assembles the final string with sign and separator, but suppress
	 * the ISO-4217 currency code.
	 *
	 * @param {String} sign The amount sign: "+" or "-".
	 * @param {String} q The formatted quantity (unsigned).
	 *
	 * @returns {String} The final formatted string.
	 */
	this._formatAsInternationalCurrencyWithNoSym = function (sign, q) {
		
		// assemble the final formatted amount by going over all possible value combinations of:
		// sign {+,-} , sign position {0,1,2,3,4} , separator {0,1,2} , symbol position {0,1}
		
		if (sign == "+") {
			
			// parentheses
			if      (this.lc.int_p_sign_posn === 0) {
				return "(" + q + ")";
			}
			
			// sign before q + sym
			else if (this.lc.int_p_sign_posn === 1 && this.lc.int_p_sep_by_space === 0 && this.lc.int_p_cs_precedes === 0) {
				return this.lc.positive_sign + q;
			}
			else if (this.lc.int_p_sign_posn === 1 && this.lc.int_p_sep_by_space === 0 && this.lc.int_p_cs_precedes === 1) {
				return this.lc.positive_sign + q;
			}
			else if (this.lc.int_p_sign_posn === 1 && this.lc.int_p_sep_by_space === 1 && this.lc.int_p_cs_precedes === 0) {
				return this.lc.positive_sign + q;
			}
			else if (this.lc.int_p_sign_posn === 1 && this.lc.int_p_sep_by_space === 1 && this.lc.int_p_cs_precedes === 1) {
				return this.lc.positive_sign + this.intSep + q;
			}
			else if (this.lc.int_p_sign_posn === 1 && this.lc.int_p_sep_by_space === 2 && this.lc.int_p_cs_precedes === 0) {
				return this.lc.positive_sign + this.intSep + q;
			}
			else if (this.lc.int_p_sign_posn === 1 && this.lc.int_p_sep_by_space === 2 && this.lc.int_p_cs_precedes === 1) {
				return this.lc.positive_sign + this.intSep + q;
			}
			
			// sign after q + sym
			else if (this.lc.int_p_sign_posn === 2 && this.lc.int_p_sep_by_space === 0 && this.lc.int_p_cs_precedes === 0) {
				return q + this.lc.positive_sign;
			}
			else if (this.lc.int_p_sign_posn === 2 && this.lc.int_p_sep_by_space === 0 && this.lc.int_p_cs_precedes === 1) {
				return q + this.lc.positive_sign;
			}
			else if (this.lc.int_p_sign_posn === 2 && this.lc.int_p_sep_by_space === 1 && this.lc.int_p_cs_precedes === 0) {
				return  q + this.intSep + this.lc.positive_sign;
			}
			else if (this.lc.int_p_sign_posn === 2 && this.lc.int_p_sep_by_space === 1 && this.lc.int_p_cs_precedes === 1) {
				return q + this.lc.positive_sign;
			}
			else if (this.lc.int_p_sign_posn === 2 && this.lc.int_p_sep_by_space === 2 && this.lc.int_p_cs_precedes === 0) {
				return q + this.intSep + this.lc.positive_sign;
			}
			else if (this.lc.int_p_sign_posn === 2 && this.lc.int_p_sep_by_space === 2 && this.lc.int_p_cs_precedes === 1) {
				return q + this.intSep + this.lc.positive_sign;
			}
			
			// sign before sym
			else if (this.lc.int_p_sign_posn === 3 && this.lc.int_p_sep_by_space === 0 && this.lc.int_p_cs_precedes === 0) {
				return q + this.lc.positive_sign;
			}
			else if (this.lc.int_p_sign_posn === 3 && this.lc.int_p_sep_by_space === 0 && this.lc.int_p_cs_precedes === 1) {
				return this.lc.positive_sign + q;
			}
			else if (this.lc.int_p_sign_posn === 3 && this.lc.int_p_sep_by_space === 1 && this.lc.int_p_cs_precedes === 0) {
				return q + this.intSep + this.lc.positive_sign;
			}
			else if (this.lc.int_p_sign_posn === 3 && this.lc.int_p_sep_by_space === 1 && this.lc.int_p_cs_precedes === 1) {
				return this.lc.positive_sign + this.intSep + q;
			}
			else if (this.lc.int_p_sign_posn === 3 && this.lc.int_p_sep_by_space === 2 && this.lc.int_p_cs_precedes === 0) {
				return q + this.lc.positive_sign;
			}
			else if (this.lc.int_p_sign_posn === 3 && this.lc.int_p_sep_by_space === 2 && this.lc.int_p_cs_precedes === 1) {
				return this.lc.positive_sign + this.intSep + q;
			}
			
			// sign after symbol
			else if (this.lc.int_p_sign_posn === 4 && this.lc.int_p_sep_by_space === 0 && this.lc.int_p_cs_precedes === 0) {
				return q + this.lc.positive_sign;
			}
			else if (this.lc.int_p_sign_posn === 4 && this.lc.int_p_sep_by_space === 0 && this.lc.int_p_cs_precedes === 1) {
				return this.lc.positive_sign + q;
			}
			else if (this.lc.int_p_sign_posn === 4 && this.lc.int_p_sep_by_space === 1 && this.lc.int_p_cs_precedes === 0) {
				return  q + this.intSep + this.lc.positive_sign;
			}
			else if (this.lc.int_p_sign_posn === 4 && this.lc.int_p_sep_by_space === 1 && this.lc.int_p_cs_precedes === 1) {
				return this.lc.positive_sign + this.intSep + q;
			}
			else if (this.lc.int_p_sign_posn === 4 && this.lc.int_p_sep_by_space === 2 && this.lc.int_p_cs_precedes === 0) {
				return q + this.intSep + this.lc.positive_sign;
			}
			else if (this.lc.int_p_sign_posn === 4 && this.lc.int_p_sep_by_space === 2 && this.lc.int_p_cs_precedes === 1) {
				return this.lc.positive_sign + q;
			}
			
		}
		else if (sign == "-") {
			
			// parentheses enclose q + sym
			if      (this.lc.int_n_sign_posn === 0) {
				return "(" + q + ")";
			}
			
			// sign before q + sym
			else if (this.lc.int_n_sign_posn === 1 && this.lc.int_n_sep_by_space === 0 && this.lc.int_n_cs_precedes === 0) {
				return this.lc.negative_sign + q;
			}
			else if (this.lc.int_n_sign_posn === 1 && this.lc.int_n_sep_by_space === 0 && this.lc.int_n_cs_precedes === 1) {
				return this.lc.negative_sign + q;
			}
			else if (this.lc.int_n_sign_posn === 1 && this.lc.int_n_sep_by_space === 1 && this.lc.int_n_cs_precedes === 0) {
				return this.lc.negative_sign + q;
			}
			else if (this.lc.int_n_sign_posn === 1 && this.lc.int_n_sep_by_space === 1 && this.lc.int_n_cs_precedes === 1) {
				return this.lc.negative_sign + this.intSep + q;
			}
			else if (this.lc.int_n_sign_posn === 1 && this.lc.int_n_sep_by_space === 2 && this.lc.int_n_cs_precedes === 0) {
				return this.lc.negative_sign + this.intSep + q;
			}
			else if (this.lc.int_n_sign_posn === 1 && this.lc.int_n_sep_by_space === 2 && this.lc.int_n_cs_precedes === 1) {
				return this.lc.negative_sign + this.intSep + q;
			}
			
			// sign after q + sym
			else if (this.lc.int_n_sign_posn === 2 && this.lc.int_n_sep_by_space === 0 && this.lc.int_n_cs_precedes === 0) {
				return q + this.lc.negative_sign;
			}
			else if (this.lc.int_n_sign_posn === 2 && this.lc.int_n_sep_by_space === 0 && this.lc.int_n_cs_precedes === 1) {
				return q + this.lc.negative_sign;
			}
			else if (this.lc.int_n_sign_posn === 2 && this.lc.int_n_sep_by_space === 1 && this.lc.int_n_cs_precedes === 0) {
				return  q + this.intSep + this.lc.negative_sign;
			}
			else if (this.lc.int_n_sign_posn === 2 && this.lc.int_n_sep_by_space === 1 && this.lc.int_n_cs_precedes === 1) {
				return q + this.lc.negative_sign;
			}
			else if (this.lc.int_n_sign_posn === 2 && this.lc.int_n_sep_by_space === 2 && this.lc.int_n_cs_precedes === 0) {
				return q + this.intSep + this.lc.negative_sign;
			}
			else if (this.lc.int_n_sign_posn === 2 && this.lc.int_n_sep_by_space === 2 && this.lc.int_n_cs_precedes === 1) {
				return q + this.intSep + this.lc.negative_sign;
			}
			
			// sign before sym
			else if (this.lc.int_n_sign_posn === 3 && this.lc.int_n_sep_by_space === 0 && this.lc.int_n_cs_precedes === 0) {
				return q + this.lc.negative_sign;
			}
			else if (this.lc.int_n_sign_posn === 3 && this.lc.int_n_sep_by_space === 0 && this.lc.int_n_cs_precedes === 1) {
				return this.lc.negative_sign + q;
			}
			else if (this.lc.int_n_sign_posn === 3 && this.lc.int_n_sep_by_space === 1 && this.lc.int_n_cs_precedes === 0) {
				return q + this.intSep + this.lc.negative_sign;
			}
			else if (this.lc.int_n_sign_posn === 3 && this.lc.int_n_sep_by_space === 1 && this.lc.int_n_cs_precedes === 1) {
				return this.lc.negative_sign + this.intSep + q;
			}
			else if (this.lc.int_n_sign_posn === 3 && this.lc.int_n_sep_by_space === 2 && this.lc.int_n_cs_precedes === 0) {
				return q + this.lc.negative_sign;
			}
			else if (this.lc.int_n_sign_posn === 3 && this.lc.int_n_sep_by_space === 2 && this.lc.int_n_cs_precedes === 1) {
				return this.lc.negative_sign + this.intSep + q;
			}
			
			// sign after symbol
			else if (this.lc.int_n_sign_posn === 4 && this.lc.int_n_sep_by_space === 0 && this.lc.int_n_cs_precedes === 0) {
				return q + this.lc.negative_sign;
			}
			else if (this.lc.int_n_sign_posn === 4 && this.lc.int_n_sep_by_space === 0 && this.lc.int_n_cs_precedes === 1) {
				return this.lc.negative_sign + q;
			}
			else if (this.lc.int_n_sign_posn === 4 && this.lc.int_n_sep_by_space === 1 && this.lc.int_n_cs_precedes === 0) {
				return  q + this.intSep + this.lc.negative_sign;
			}
			else if (this.lc.int_n_sign_posn === 4 && this.lc.int_n_sep_by_space === 1 && this.lc.int_n_cs_precedes === 1) {
				return this.lc.negative_sign + this.intSep + q;
			}
			else if (this.lc.int_n_sign_posn === 4 && this.lc.int_n_sep_by_space === 2 && this.lc.int_n_cs_precedes === 0) {
				return q + this.intSep + this.lc.negative_sign;
			}
			else if (this.lc.int_n_sign_posn === 4 && this.lc.int_n_sep_by_space === 2 && this.lc.int_n_cs_precedes === 1) {
				return this.lc.negative_sign + q;
			}
		}
		
		// throw error if we fall through
		throw "Error: Invalid POSIX LC_MONETARY definition";
	};
};


/** 
 * @class 
 * Class for parsing localised number strings.
 *
 * @public
 * @constructor 
 * @description Creates a new numeric parser for the specified locale.
 *
 * @param {jsworld.Locale} locale A locale object specifying the required 
 *        POSIX LC_NUMERIC formatting properties.
 *
 * @throws Error on constructor failure.
 */
jsworld.NumericParser = function(locale) {

	if (typeof locale != "object" || locale._className != "jsworld.Locale")
		throw "Constructor error: You must provide a valid jsworld.Locale instance";

	this.lc = locale;
	
	
	/**
	 * @public
	 *
	 * @description Parses a numeric string formatted according to the 
	 * preset locale. Leading and trailing whitespace is ignored; the number
	 * may also be formatted without thousands separators.
	 *
	 * @param {String} formattedNumber The formatted number.
	 *
	 * @returns {Number} The parsed number.
	 *
	 * @throws Error on a parse exception.
	 */
	this.parse = function(formattedNumber) {
	
		if (typeof formattedNumber != "string")
			throw "Parse error: Argument must be a string";
	
		// trim whitespace
		var s = jsworld._trim(formattedNumber);
	
		// remove any thousand separator symbols
		s = jsworld._stringReplaceAll(formattedNumber, this.lc.thousands_sep, "");
		
		// replace any local decimal point symbols with the symbol used
		// in JavaScript "."
		s = jsworld._stringReplaceAll(s, this.lc.decimal_point, ".");
		
		// test if the string represents a number
		if (jsworld._isNumber(s))
			return parseFloat(s, 10);		
		else
			throw "Parse error: Invalid number string";
	};
};


/** 
 * @class 
 * Class for parsing localised date and time strings.
 *
 * @public
 * @constructor 
 * @description Creates a new date/time parser for the specified locale.
 *
 * @param {jsworld.Locale} locale A locale object specifying the required 
 *        POSIX LC_TIME formatting properties.
 *
 * @throws Error on constructor failure.
 */
jsworld.DateTimeParser = function(locale) {

	if (typeof locale != "object" || locale._className != "jsworld.Locale")
		throw "Constructor error: You must provide a valid jsworld.Locale instance.";

	this.lc = locale;

	
	/**
	 * @public
	 *
	 * @description Parses a time string formatted according to the 
	 * POSIX LC_TIME t_fmt property of the preset locale.
	 *
	 * @param {String} formattedTime The formatted time.
	 *
	 * @returns {String} The parsed time in ISO-8601 format (HH:MM:SS), e.g.
	 *          "23:59:59".
	 *
	 * @throws Error on a parse exception.
	 */
	this.parseTime = function(formattedTime) {
	
		if (typeof formattedTime != "string")
			throw "Parse error: Argument must be a string";
	
		var dt = this._extractTokens(this.lc.t_fmt, formattedTime);
		
		var timeDefined = false;
		
		if (dt.hour !== null && dt.minute !== null && dt.second !== null) {
			timeDefined = true;
		}
		else if (dt.hourAmPm !== null && dt.am !== null && dt.minute !== null && dt.second !== null) {
			if (dt.am) {
				// AM [12(midnight), 1 .. 11]
				if (dt.hourAmPm == 12)
					dt.hour = 0;
				else
					dt.hour = parseInt(dt.hourAmPm, 10);
			}
			else {
				// PM [12(noon), 1 .. 11]
				if (dt.hourAmPm == 12)
					dt.hour = 12;
				else
					dt.hour = parseInt(dt.hourAmPm, 10) + 12;
			}
			timeDefined = true;
		}
		
		if (timeDefined)
			return jsworld._zeroPad(dt.hour, 2) + 
			       ":" + 
			       jsworld._zeroPad(dt.minute, 2) + 
			       ":" + 
			       jsworld._zeroPad(dt.second, 2);
		else
			throw "Parse error: Invalid/ambiguous time string";
	};
	
	
	/**
	 * @public
	 *
	 * @description Parses a date string formatted according to the 
	 * POSIX LC_TIME d_fmt property of the preset locale.
	 *
	 * @param {String} formattedDate The formatted date, must be valid.
	 *
	 * @returns {String} The parsed date in ISO-8601 format (YYYY-MM-DD), 
	 *          e.g. "2010-03-31".
	 *
	 * @throws Error on a parse exception.
	 */
	this.parseDate = function(formattedDate) {
	
		if (typeof formattedDate != "string")
			throw "Parse error: Argument must be a string";
	
		var dt = this._extractTokens(this.lc.d_fmt, formattedDate);
		
		var dateDefined = false;
		
		if (dt.year !== null && dt.month !== null && dt.day !== null) {
			dateDefined = true;
		}
		
		if (dateDefined)
			return jsworld._zeroPad(dt.year, 4) + 
			       "-" + 
			       jsworld._zeroPad(dt.month, 2) + 
			       "-" + 
			       jsworld._zeroPad(dt.day, 2);
		else
			throw "Parse error: Invalid date string";
	};
	
	
	/**
	 * @public
	 *
	 * @description Parses a date/time string formatted according to the 
	 * POSIX LC_TIME d_t_fmt property of the preset locale.
	 *
	 * @param {String} formattedDateTime The formatted date/time, must be
	 *        valid.
	 *
	 * @returns {String} The parsed date/time in ISO-8601 format 
	 *          (YYYY-MM-DD HH:MM:SS), e.g. "2010-03-31 23:59:59".
	 *
	 * @throws Error on a parse exception.
	 */
	this.parseDateTime = function(formattedDateTime) {
	
		if (typeof formattedDateTime != "string")
			throw "Parse error: Argument must be a string";
		
		var dt = this._extractTokens(this.lc.d_t_fmt, formattedDateTime);
		
		var timeDefined = false;
		var dateDefined = false;
	
		if (dt.hour !== null && dt.minute !== null && dt.second !== null) {
			timeDefined = true;
		}
		else if (dt.hourAmPm !== null && dt.am !== null && dt.minute !== null && dt.second !== null) {
			if (dt.am) {
				// AM [12(midnight), 1 .. 11]
				if (dt.hourAmPm == 12)
					dt.hour = 0;
				else
					dt.hour = parseInt(dt.hourAmPm, 10);
			}
			else {
				// PM [12(noon), 1 .. 11]
				if (dt.hourAmPm == 12)
					dt.hour = 12;
				else
					dt.hour = parseInt(dt.hourAmPm, 10) + 12;
			}
			timeDefined = true;
		}
		
		if (dt.year !== null && dt.month !== null && dt.day !== null) {
			dateDefined = true;
		}
		
		if (dateDefined && timeDefined)
			return jsworld._zeroPad(dt.year, 4) + 
			       "-" + 
			       jsworld._zeroPad(dt.month, 2) + 
			       "-" + 
			       jsworld._zeroPad(dt.day, 2) + 
			       " " +
			       jsworld._zeroPad(dt.hour, 2) + 
			       ":" + 
			       jsworld._zeroPad(dt.minute, 2) + 
			       ":" + 
			       jsworld._zeroPad(dt.second, 2);
		else
			throw "Parse error: Invalid/ambiguous date/time string";
	};
	
	
	/**
	 * @private
	 *
	 * @description Parses a string according to the specified format
	 * specification.
	 *
	 * @param {String} fmtSpec The format specification, e.g. "%I:%M:%S %p".
	 * @param {String} s The string to parse.
	 *
	 * @returns {object} An object with set properties year, month, day,
	 *          hour, minute and second if the corresponding values are
	 *          found in the parsed string.
	 *
	 * @throws Error on a parse exception.
	 */
	this._extractTokens = function(fmtSpec, s) {
	
		// the return object containing the parsed date/time properties
		var dt = {
			// for date and date/time strings
			"year"     : null,
			"month"    : null,
			"day"      : null,
			
			// for time and date/time strings
			"hour"     : null,
			"hourAmPm" : null,
			"am"       : null,
			"minute"   : null,
			"second"   : null,
			
			// used internally only
			"weekday"  : null
		};

	
		// extract and process each token in the date/time spec
		while (fmtSpec.length > 0) {
		
			// Do we have a valid "%\w" placeholder in stream?
			if (fmtSpec.charAt(0) == "%" && fmtSpec.charAt(1) != "") {
				
				// get placeholder
				var placeholder = fmtSpec.substring(0,2);
				
				if (placeholder == "%%") {
					// escaped '%''
					s = s.substring(1);
				}
				else if (placeholder == "%a") {
					// abbreviated weekday name
					for (var i = 0; i < this.lc.abday.length; i++) {
					
						if (jsworld._stringStartsWith(s, this.lc.abday[i])) {
							dt.weekday = i;
							s = s.substring(this.lc.abday[i].length);
							break;
						}
					}
					
					if (dt.weekday === null)
						throw "Parse error: Unrecognised abbreviated weekday name (%a)";
				}
				else if (placeholder == "%A") {
					// weekday name
					for (var i = 0; i < this.lc.day.length; i++) {
					
						if (jsworld._stringStartsWith(s, this.lc.day[i])) {
							dt.weekday = i;
							s = s.substring(this.lc.day[i].length);
							break;
						}
					}
					
					if (dt.weekday === null)
						throw "Parse error: Unrecognised weekday name (%A)";
				}
				else if (placeholder == "%b" || placeholder == "%h") {
					// abbreviated month name
					for (var i = 0; i < this.lc.abmon.length; i++) {
			
						if (jsworld._stringStartsWith(s, this.lc.abmon[i])) {
							dt.month = i + 1;
							s = s.substring(this.lc.abmon[i].length);
							break;
						}
					}

					if (dt.month === null)
						throw "Parse error: Unrecognised abbreviated month name (%b)";
				}
				else if (placeholder == "%B") {
					// month name
					for (var i = 0; i < this.lc.mon.length; i++) {
			
						if (jsworld._stringStartsWith(s, this.lc.mon[i])) {
							dt.month = i + 1;
							s = s.substring(this.lc.mon[i].length);
							break;
						}
					}

					if (dt.month === null)
						throw "Parse error: Unrecognised month name (%B)";
				}
				else if (placeholder == "%d") {
					// day of the month [01..31]
					if (/^0[1-9]|[1-2][0-9]|3[0-1]/.test(s)) {
						dt.day = parseInt(s.substring(0,2), 10);
						s = s.substring(2);
					}
					else	
						throw "Parse error: Unrecognised day of the month (%d)";
				}
				else if (placeholder == "%e") {
					// day of the month [1..31]
					
					// Note: if %e is leading in fmt string -> space padded!
					
					var day = s.match(/^\s?(\d{1,2})/);
					dt.day = parseInt(day, 10);
					
					if (isNaN(dt.day) || dt.day < 1 || dt.day > 31)
						throw "Parse error: Unrecognised day of the month (%e)";
					
					s = s.substring(day.length);
				}
				else if (placeholder == "%F") {
					// equivalent to %Y-%m-%d (ISO-8601 date format)
					
					// year [nnnn]
					if (/^\d\d\d\d/.test(s)) {
						dt.year = parseInt(s.substring(0,4), 10);
						s = s.substring(4);
					}
					else {
						throw "Parse error: Unrecognised date (%F)";
					}
					
					// -
					if (jsworld._stringStartsWith(s, "-"))
						s = s.substring(1);
					else
						throw "Parse error: Unrecognised date (%F)";
					
					// month [01..12]
					if (/^0[1-9]|1[0-2]/.test(s)) {
						dt.month = parseInt(s.substring(0,2), 10);
						s = s.substring(2);
					}
					else	
						throw "Parse error: Unrecognised date (%F)";

					// -
					if (jsworld._stringStartsWith(s, "-"))
						s = s.substring(1);
					else
						throw "Parse error: Unrecognised date (%F)";
					
					// day of the month [01..31]
					if (/^0[1-9]|[1-2][0-9]|3[0-1]/.test(s)) {
						dt.day = parseInt(s.substring(0,2), 10);
						s = s.substring(2);
					}
					else	
						throw "Parse error: Unrecognised date (%F)";
				}
				else if (placeholder == "%H") {
					// hour [00..23]
					if (/^[0-1][0-9]|2[0-3]/.test(s)) {
						dt.hour = parseInt(s.substring(0,2), 10);
						s = s.substring(2);
					}
					else
						throw "Parse error: Unrecognised hour (%H)";
				}
				else if (placeholder == "%I") {
					// hour [01..12]
					if (/^0[1-9]|1[0-2]/.test(s)) {
						dt.hourAmPm = parseInt(s.substring(0,2), 10);
						s = s.substring(2);
					}
					else
						throw "Parse error: Unrecognised hour (%I)";
				}
				else if (placeholder == "%k") {
					// hour [0..23]
					var h = s.match(/^(\d{1,2})/);
					dt.hour = parseInt(h, 10);
					
					if (isNaN(dt.hour) || dt.hour < 0 || dt.hour > 23)
						throw "Parse error: Unrecognised hour (%k)";
					
					s = s.substring(h.length);
				}
				else if (placeholder == "%l") {
					// hour AM/PM [1..12]
					var h = s.match(/^(\d{1,2})/);
					dt.hourAmPm = parseInt(h, 10);
					
					if (isNaN(dt.hourAmPm) || dt.hourAmPm < 1 || dt.hourAmPm > 12)
						throw "Parse error: Unrecognised hour (%l)";
					
					s = s.substring(h.length);
				}
				else if (placeholder == "%m") {
					// month [01..12]
					if (/^0[1-9]|1[0-2]/.test(s)) {
						dt.month = parseInt(s.substring(0,2), 10);
						s = s.substring(2);
					}
					else	
						throw "Parse error: Unrecognised month (%m)";
				}
				else if (placeholder == "%M") {
					// minute [00..59]
					if (/^[0-5][0-9]/.test(s)) {
						dt.minute = parseInt(s.substring(0,2), 10);
						s = s.substring(2);
					}
					else	
						throw "Parse error: Unrecognised minute (%M)";
				}
				else if (placeholder == "%n") {
					// new line
					
					if (s.charAt(0) == "\n")
						s = s.substring(1);
					else
						throw "Parse error: Unrecognised new line (%n)";
				}
				else if (placeholder == "%p") {
					// locale's equivalent of AM/PM
					if (jsworld._stringStartsWith(s, this.lc.am)) {
						dt.am = true;
						s = s.substring(this.lc.am.length);
					}
					else if (jsworld._stringStartsWith(s, this.lc.pm)) {
						dt.am = false;
						s = s.substring(this.lc.pm.length);
					}
					else
						throw "Parse error: Unrecognised AM/PM value (%p)";
				}
				else if (placeholder == "%P") {
					// same as %p but forced lower case
					if (jsworld._stringStartsWith(s, this.lc.am.toLowerCase())) {
						dt.am = true;
						s = s.substring(this.lc.am.length);
					}
					else if (jsworld._stringStartsWith(s, this.lc.pm.toLowerCase())) {
						dt.am = false;
						s = s.substring(this.lc.pm.length);
					}
					else
						throw "Parse error: Unrecognised AM/PM value (%P)";
				}
				else if (placeholder == "%R") {
					// same as %H:%M
					
					// hour [00..23]
					if (/^[0-1][0-9]|2[0-3]/.test(s)) {
						dt.hour = parseInt(s.substring(0,2), 10);
						s = s.substring(2);
					}
					else
						throw "Parse error: Unrecognised time (%R)";
					
					// :
					if (jsworld._stringStartsWith(s, ":"))
						s = s.substring(1);
					else
						throw "Parse error: Unrecognised time (%R)";

					// minute [00..59]
					if (/^[0-5][0-9]/.test(s)) {
						dt.minute = parseInt(s.substring(0,2), 10);
						s = s.substring(2);
					}
					else	
						throw "Parse error: Unrecognised time (%R)";

				}
				else if (placeholder == "%S") {
					// second [00..59]
					if (/^[0-5][0-9]/.test(s)) {
						dt.second = parseInt(s.substring(0,2), 10);
						s = s.substring(2);
					}
					else
						throw "Parse error: Unrecognised second (%S)";
				}
				else if (placeholder == "%T") {
					// same as %H:%M:%S
					
					// hour [00..23]
					if (/^[0-1][0-9]|2[0-3]/.test(s)) {
						dt.hour = parseInt(s.substring(0,2), 10);
						s = s.substring(2);
					}
					else
						throw "Parse error: Unrecognised time (%T)";
					
					// :
					if (jsworld._stringStartsWith(s, ":"))
						s = s.substring(1);
					else
						throw "Parse error: Unrecognised time (%T)";

					// minute [00..59]
					if (/^[0-5][0-9]/.test(s)) {
						dt.minute = parseInt(s.substring(0,2), 10);
						s = s.substring(2);
					}
					else	
						throw "Parse error: Unrecognised time (%T)";
					
					// :
					if (jsworld._stringStartsWith(s, ":"))
						s = s.substring(1);
					else
						throw "Parse error: Unrecognised time (%T)";
					
					// second [00..59]
					if (/^[0-5][0-9]/.test(s)) {
						dt.second = parseInt(s.substring(0,2), 10);
						s = s.substring(2);
					}
					else
						throw "Parse error: Unrecognised time (%T)";
				}
				else if (placeholder == "%w") {
					// weekday [0..6]
					if (/^\d/.test(s)) {
						dt.weekday = parseInt(s.substring(0,1), 10);
						s = s.substring(1);
					}
					else 
						throw "Parse error: Unrecognised weekday number (%w)";
				}
				else if (placeholder == "%y") {
					// year [00..99]
					if (/^\d\d/.test(s)) {
						var year2digits = parseInt(s.substring(0,2), 10);
						
						// this conversion to year[nnnn] is arbitrary!!!
						if (year2digits > 50)
							dt.year = 1900 + year2digits;
						else
							dt.year = 2000 + year2digits;
						
						s = s.substring(2);
					}
					else
						throw "Parse error: Unrecognised year (%y)";
				}
				else if (placeholder == "%Y") {
					// year [nnnn]
					if (/^\d\d\d\d/.test(s)) {
						dt.year = parseInt(s.substring(0,4), 10);
						s = s.substring(4);
					}
					else
						throw "Parse error: Unrecognised year (%Y)";
				}
				
				else if (placeholder == "%Z") {
					// time-zone place holder is not supported
					
					if (fmtSpec.length === 0)
						break; // ignore rest of fmt spec
				}

				// remove the spec placeholder that was just parsed
				fmtSpec = fmtSpec.substring(2);
			}
			else {
				// If we don't have a placeholder, the chars
				// at pos. 0 of format spec and parsed string must match
				
				// Note: Space chars treated 1:1 !
				
				if (fmtSpec.charAt(0) != s.charAt(0))
					throw "Parse error: Unexpected symbol \"" + s.charAt(0) + "\" in date/time string";
			
				fmtSpec = fmtSpec.substring(1);
				s = s.substring(1);
			}
		}
		
		// parsing finished, return composite date/time object
		return dt;
	};
};


/** 
 * @class 
 * Class for parsing localised currency amount strings.
 *
 * @public
 * @constructor 
 * @description Creates a new monetary parser for the specified locale.
 *
 * @param {jsworld.Locale} locale A locale object specifying the required 
 *        POSIX LC_MONETARY formatting properties.
 *
 * @throws Error on constructor failure.
 */
jsworld.MonetaryParser = function(locale) {

	if (typeof locale != "object" || locale._className != "jsworld.Locale")
		throw "Constructor error: You must provide a valid jsworld.Locale instance";


	this.lc = locale;
	
	
	/**
	 * @public
	 *
	 * @description Parses a currency amount string formatted according to 
	 * the preset locale. Leading and trailing whitespace is ignored; the 
	 * amount may also be formatted without thousands separators. Both
	 * the local (shorthand) symbol and the ISO 4217 code are accepted to 
	 * designate the currency in the formatted amount.
	 *
	 * @param {String} formattedCurrency The formatted currency amount.
	 *
	 * @returns {Number} The parsed amount.
	 *
	 * @throws Error on a parse exception.
	 */
	this.parse = function(formattedCurrency) {
	
		if (typeof formattedCurrency != "string")
			throw "Parse error: Argument must be a string";
	
		// Detect the format type and remove the currency symbol
		var symbolType = this._detectCurrencySymbolType(formattedCurrency);
	
		var formatType, s;
	
		if (symbolType == "local") {
			formatType = "local";
			s = formattedCurrency.replace(this.lc.getCurrencySymbol(), "");
		}
		else if (symbolType == "int") {
			formatType = "int";
			s = formattedCurrency.replace(this.lc.getIntCurrencySymbol(), "");
		}
		else if (symbolType == "none") {
			formatType = "local"; // assume local
			s = formattedCurrency;
		}
		else
			throw "Parse error: Internal assert failure";
		
		// Remove any thousands separators
		s = jsworld._stringReplaceAll(s, this.lc.mon_thousands_sep, "");
		
		// Replace any local radix char with JavaScript's "."
		s = s.replace(this.lc.mon_decimal_point, ".");
		
		// Remove all whitespaces
		s = s.replace(/\s*/g, "");
		
		// Remove any local non-negative sign
		s = this._removeLocalNonNegativeSign(s, formatType);
		
		// Replace any local minus sign with JavaScript's "-" and put
		// it in front of the amount if necessary
		// (special parentheses rule checked too)
		s = this._normaliseNegativeSign(s, formatType);
		
		// Finally, we should be left with a bare parsable decimal number
		if (jsworld._isNumber(s))
			return parseFloat(s, 10);
		else
			throw "Parse error: Invalid currency amount string";
	};
	
	
	/**
	 * @private
	 *
	 * @description Tries to detect the symbol type used in the specified
	 *              formatted currency string: local(shorthand), 
	 *              international (ISO-4217 code) or none.
	 *
	 * @param {String} formattedCurrency The the formatted currency string.
	 *
	 * @return {String} With possible values "local", "int" or "none".
	 */
	this._detectCurrencySymbolType = function(formattedCurrency) {
	
		// Check for whichever sign (int/local) is longer first
		// to cover cases such as MOP/MOP$ and ZAR/R
		
		if (this.lc.getCurrencySymbol().length > this.lc.getIntCurrencySymbol().length) {
		
			if (formattedCurrency.indexOf(this.lc.getCurrencySymbol()) != -1)
				return "local";
			else if (formattedCurrency.indexOf(this.lc.getIntCurrencySymbol()) != -1)
				return "int";
			else
				return "none";
		}
		else {
			if (formattedCurrency.indexOf(this.lc.getIntCurrencySymbol()) != -1)
				return "int";
			else if (formattedCurrency.indexOf(this.lc.getCurrencySymbol()) != -1)
				return "local";
			else
				return "none";
		}
	};
	
	
	/**
	 * @private
	 *
	 * @description Removes a local non-negative sign in a formatted 
	 * currency string if it is found. This is done according to the
	 * locale properties p_sign_posn and int_p_sign_posn.
	 *
	 * @param {String} s The input string.
	 * @param {String} formatType With possible values "local" or "int".
	 *
	 * @returns {String} The processed string.
	 */
	this._removeLocalNonNegativeSign = function(s, formatType) {
	
		s = s.replace(this.lc.positive_sign, "");
	
		// check for enclosing parentheses rule
		if (((formatType == "local" && this.lc.p_sign_posn     === 0) ||
		     (formatType == "int"   && this.lc.int_p_sign_posn === 0)    ) &&
		      /\(\d+\.?\d*\)/.test(s)) {
			s = s.replace("(", "");
			s = s.replace(")", "");
		}
		
		return s;
	};
	
	
	/**
	 * @private
	 *
	 * @description Replaces a local negative sign with the standard
	 * JavaScript minus ("-") sign placed in the correct position 
	 * (preceding the amount). This is done according to the locale
	 * properties for negative sign symbol and relative position.
	 *
	 * @param {String} s The input string.
	 * @param {String} formatType With possible values "local" or "int".
	 *
	 * @returns {String} The processed string.
	 */
	this._normaliseNegativeSign = function(s, formatType) {
	
		// replace local negative symbol with JavaScript's "-"
		s = s.replace(this.lc.negative_sign, "-");
	
		// check for enclosing parentheses rule and replace them
		// with negative sign before the amount
		if ((formatType == "local" && this.lc.n_sign_posn     === 0) ||
		    (formatType == "int"   && this.lc.int_n_sign_posn === 0)    ) {
		    
			if (/^\(\d+\.?\d*\)$/.test(s)) {
		     
				s = s.replace("(", "");
				s = s.replace(")", "");
				return "-" + s;
			}
		}
		
		// check for rule negative sign succeeding the amount
		if (formatType == "local" && this.lc.n_sign_posn     == 2 ||
		    formatType == "int"   && this.lc.int_n_sign_posn == 2   ) {
		
			if (/^\d+\.?\d*-$/.test(s)) {
				s = s.replace("-", "");
				return "-" + s;
			}
		}
	
		// check for rule cur. sym. succeeds and sign adjacent
		if (formatType == "local" && this.lc.n_cs_precedes     === 0 && this.lc.n_sign_posn     == 3 ||
		    formatType == "local" && this.lc.n_cs_precedes     === 0 && this.lc.n_sign_posn     == 4 ||
		    formatType == "int"   && this.lc.int_n_cs_precedes === 0 && this.lc.int_n_sign_posn == 3 ||
		    formatType == "int"   && this.lc.int_n_cs_precedes === 0 && this.lc.int_n_sign_posn == 4    ) {
		    
		    	if (/^\d+\.?\d*-$/.test(s)) {
				s = s.replace("-", "");
				return "-" + s;
			}
		}
		
		return s;
	};
};

// end-of-file
