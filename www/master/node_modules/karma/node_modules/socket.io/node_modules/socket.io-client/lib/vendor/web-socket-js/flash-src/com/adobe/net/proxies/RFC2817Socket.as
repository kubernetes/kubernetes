/*
	Adobe Systems Incorporated(r) Source Code License Agreement
	Copyright(c) 2005 Adobe Systems Incorporated. All rights reserved.
	
	Please read this Source Code License Agreement carefully before using
	the source code.
	
	Adobe Systems Incorporated grants to you a perpetual, worldwide, non-exclusive, 
	no-charge, royalty-free, irrevocable copyright license, to reproduce,
	prepare derivative works of, publicly display, publicly perform, and
	distribute this source code and such derivative works in source or 
	object code form without any attribution requirements.  
	
	The name "Adobe Systems Incorporated" must not be used to endorse or promote products
	derived from the source code without prior written permission.
	
	You agree to indemnify, hold harmless and defend Adobe Systems Incorporated from and
	against any loss, damage, claims or lawsuits, including attorney's 
	fees that arise or result from your use or distribution of the source 
	code.
	
	THIS SOURCE CODE IS PROVIDED "AS IS" AND "WITH ALL FAULTS", WITHOUT 
	ANY TECHNICAL SUPPORT OR ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
	BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
	FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  ALSO, THERE IS NO WARRANTY OF 
	NON-INFRINGEMENT, TITLE OR QUIET ENJOYMENT.  IN NO EVENT SHALL MACROMEDIA
	OR ITS SUPPLIERS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
	EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
	PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
	OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
	WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
	OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOURCE CODE, EVEN IF
	ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
package com.adobe.net.proxies
{
	import flash.events.Event;
	import flash.events.IOErrorEvent;
	import flash.events.ProgressEvent;
	import flash.net.Socket;

	/**
	 * This class allows TCP socket connections through HTTP proxies in accordance with
	 * RFC 2817:
	 * 
	 * ftp://ftp.rfc-editor.org/in-notes/rfc2817.txt
	 * 
	 * It can also be used to make direct connections to a destination, as well. If you
	 * pass the host and port into the constructor, no proxy will be used. You can also
	 * call connect, passing in the host and the port, and if you didn't set the proxy
	 * info, a direct connection will be made. A proxy is only used after you have called
	 * the setProxyInfo function.
	 * 
	 * The connection to and negotiation with the proxy is completely hidden. All the
	 * same events are thrown whether you are using a proxy or not, and the data you
	 * receive from the target server will look exact as it would if you were connected
	 * to it directly rather than through a proxy.
	 * 
	 * @author Christian Cantrell
	 * 
	 **/
	public class RFC2817Socket
		extends Socket
	{
		private var proxyHost:String = null;
		private var host:String = null;
		private var proxyPort:int = 0;
		private var port:int = 0;
		private var deferredEventHandlers:Object = new Object();
		private var buffer:String = new String();

		/**
		 * Construct a new RFC2817Socket object. If you pass in the host and the port,
		 * no proxy will be used. If you want to use a proxy, instantiate with no
		 * arguments, call setProxyInfo, then call connect.
		 **/
		public function RFC2817Socket(host:String = null, port:int = 0)
		{
			if (host != null && port != 0)
			{
				super(host, port);
			}
		}
		
		/**
		 * Set the proxy host and port number. Your connection will only proxied if
		 * this function has been called.
		 **/
		public function setProxyInfo(host:String, port:int):void
		{
			this.proxyHost = host;
			this.proxyPort = port;

			var deferredSocketDataHandler:Object = this.deferredEventHandlers[ProgressEvent.SOCKET_DATA];
			var deferredConnectHandler:Object = this.deferredEventHandlers[Event.CONNECT];

			if (deferredSocketDataHandler != null)
			{
				super.removeEventListener(ProgressEvent.SOCKET_DATA, deferredSocketDataHandler.listener, deferredSocketDataHandler.useCapture);
			}

			if (deferredConnectHandler != null)
			{
				super.removeEventListener(Event.CONNECT, deferredConnectHandler.listener, deferredConnectHandler.useCapture);
			}
		}
		
		/**
		 * Connect to the specified host over the specified port. If you want your
		 * connection proxied, call the setProxyInfo function first.
		 **/
		public override function connect(host:String, port:int):void
		{
			if (this.proxyHost == null)
			{
				this.redirectConnectEvent();
				this.redirectSocketDataEvent();
				super.connect(host, port);
			}
			else
			{
				this.host = host;
				this.port = port;
				super.addEventListener(Event.CONNECT, this.onConnect);
				super.addEventListener(ProgressEvent.SOCKET_DATA, this.onSocketData);
				super.connect(this.proxyHost, this.proxyPort);
			}
		}

		private function onConnect(event:Event):void
		{
			this.writeUTFBytes("CONNECT "+this.host+":"+this.port+" HTTP/1.1\n\n");
			this.flush();
			this.redirectConnectEvent();
		}
		
		private function onSocketData(event:ProgressEvent):void
		{
			while (this.bytesAvailable != 0)
			{
				this.buffer += this.readUTFBytes(1);
				if (this.buffer.search(/\r?\n\r?\n$/) != -1)
				{
					this.checkResponse(event);
					break;
				}
			}
		}
		
		private function checkResponse(event:ProgressEvent):void
		{
			var responseCode:String = this.buffer.substr(this.buffer.indexOf(" ")+1, 3);

			if (responseCode.search(/^2/) == -1)
			{
				var ioError:IOErrorEvent = new IOErrorEvent(IOErrorEvent.IO_ERROR);
				ioError.text = "Error connecting to the proxy ["+this.proxyHost+"] on port ["+this.proxyPort+"]: " + this.buffer;
				this.dispatchEvent(ioError);
			}
			else
			{
				this.redirectSocketDataEvent();
				this.dispatchEvent(new Event(Event.CONNECT));
				if (this.bytesAvailable > 0)
				{
					this.dispatchEvent(event);
				}
			}
			this.buffer = null;
		}
		
		private function redirectConnectEvent():void
		{
			super.removeEventListener(Event.CONNECT, onConnect);
			var deferredEventHandler:Object = this.deferredEventHandlers[Event.CONNECT];
			if (deferredEventHandler != null)
			{
				super.addEventListener(Event.CONNECT, deferredEventHandler.listener, deferredEventHandler.useCapture, deferredEventHandler.priority, deferredEventHandler.useWeakReference);			
			}
		}
		
		private function redirectSocketDataEvent():void
		{
			super.removeEventListener(ProgressEvent.SOCKET_DATA, onSocketData);
			var deferredEventHandler:Object = this.deferredEventHandlers[ProgressEvent.SOCKET_DATA];
			if (deferredEventHandler != null)
			{
				super.addEventListener(ProgressEvent.SOCKET_DATA, deferredEventHandler.listener, deferredEventHandler.useCapture, deferredEventHandler.priority, deferredEventHandler.useWeakReference);			
			}
		}
		
		public override function addEventListener(type:String, listener:Function, useCapture:Boolean = false, priority:int=0.0, useWeakReference:Boolean=false):void
		{
			if (type == Event.CONNECT || type == ProgressEvent.SOCKET_DATA)
			{
				this.deferredEventHandlers[type] = {listener:listener,useCapture:useCapture, priority:priority, useWeakReference:useWeakReference};
			}
			else
			{
				super.addEventListener(type, listener, useCapture, priority, useWeakReference);
			}
		}
	}
}