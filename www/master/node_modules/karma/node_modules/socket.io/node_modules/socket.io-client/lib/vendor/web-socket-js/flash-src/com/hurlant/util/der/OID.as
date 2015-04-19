/**
 * OID
 * 
 * A list of various ObjectIdentifiers.
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.util.der
{
	public class OID
	{

		public static const RSA_ENCRYPTION:String           = "1.2.840.113549.1.1.1";
		public static const MD2_WITH_RSA_ENCRYPTION:String  = "1.2.840.113549.1.1.2";
		public static const MD5_WITH_RSA_ENCRYPTION:String  = "1.2.840.113549.1.1.4";
		public static const SHA1_WITH_RSA_ENCRYPTION:String = "1.2.840.113549.1.1.5";
		public static const MD2_ALGORITHM:String = "1.2.840.113549.2.2";
		public static const MD5_ALGORITHM:String = "1.2.840.113549.2.5";
		public static const DSA:String = "1.2.840.10040.4.1";
		public static const DSA_WITH_SHA1:String = "1.2.840.10040.4.3";
		public static const DH_PUBLIC_NUMBER:String = "1.2.840.10046.2.1";
		public static const SHA1_ALGORITHM:String = "1.3.14.3.2.26";
		
		public static const COMMON_NAME:String = "2.5.4.3";
		public static const SURNAME:String = "2.5.4.4";
		public static const COUNTRY_NAME:String = "2.5.4.6";
		public static const LOCALITY_NAME:String = "2.5.4.7";
		public static const STATE_NAME:String = "2.5.4.8";
		public static const ORGANIZATION_NAME:String = "2.5.4.10";
		public static const ORG_UNIT_NAME:String = "2.5.4.11";
		public static const TITLE:String = "2.5.4.12";

	}
}