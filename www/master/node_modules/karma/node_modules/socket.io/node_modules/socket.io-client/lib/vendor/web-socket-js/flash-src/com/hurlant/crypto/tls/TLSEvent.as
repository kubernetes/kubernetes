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
	
	public class TLSEvent extends Event {
		
		static public const DATA:String = "data";
		static public const READY:String = "ready";
		static public const PROMPT_ACCEPT_CERT:String = "promptAcceptCert";
		
		public var data:ByteArray;
		
		public function TLSEvent(type:String, data:ByteArray = null) {
			this.data = data;
			super(type, false, false);
		}
	}
}