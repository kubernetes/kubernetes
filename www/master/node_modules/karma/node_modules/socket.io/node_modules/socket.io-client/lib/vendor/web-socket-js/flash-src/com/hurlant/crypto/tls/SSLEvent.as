/**
 * SSLEvent
 * 
 * This is used by TLSEngine to let the application layer know
 * when we're ready for sending, or have received application data
 * This Event was created by Bobby Parker to support SSL 3.0.
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.tls {
	import flash.events.Event;
	import flash.utils.ByteArray;
	
	public class SSLEvent extends Event {
		
		static public const DATA:String = "data";
		static public const READY:String = "ready";
		
		public var data:ByteArray;
		
		public function SSLEvent(type:String, data:ByteArray = null) {
			this.data = data;
			super(type, false, false);
		}
	}
}