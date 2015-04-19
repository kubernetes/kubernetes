/**
 * KeyExchanges
 * 
 * An enumeration of key exchange methods defined by TLS
 * ( right now, only RSA is actually implemented )
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.tls {
	public class KeyExchanges {
		public static const NULL:uint = 0;
		public static const RSA:uint = 1;
		public static const DH_DSS:uint = 2;
		public static const DH_RSA:uint = 3;
		public static const DHE_DSS:uint = 4;
		public static const DHE_RSA:uint = 5;
		public static const DH_anon:uint = 6;
		
		public static function useRSA(p:uint):Boolean {
			return (p==RSA);
		}
	}
}