jsworld.Locale = function(properties) {
	
	// LC_NUMERIC

	
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
	
}
