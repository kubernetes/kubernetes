/**
 * TLSEvent
 * 
 * This is used by TLSEngine to let the application layer know
 * when we're ready for sending, or have received application data
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.tls {
	import flash.events.Event;
	import flash.utils.ByteArray;
	import com.hurlant.crypto.cert.X509Certificate;
	
	public class TLSSocketEvent extends Event {
		
		static public const PROMPT_ACCEPT_CERT:String = "promptAcceptCert";
		
		public var cert:X509Certificate;
		
		public function TLSSocketEvent( cert:X509Certificate = null) { 
			super(PROMPT_ACCEPT_CERT, false, false);
			this.cert = cert;
		}
	}
}