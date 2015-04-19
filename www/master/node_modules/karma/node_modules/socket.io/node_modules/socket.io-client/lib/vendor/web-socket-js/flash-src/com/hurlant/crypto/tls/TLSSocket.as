/**
 * TLSSocket
 * 
 * This is the "end-user" TLS class.
 * It works just like a Socket, by encapsulating a Socket and
 * wrapping the TLS protocol around the data that passes over it.
 * This class can either create a socket connection, or reuse an
 * existing connected socket. The later is useful for STARTTLS flows.
 * 
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.tls {
	import flash.events.Event;
	import flash.events.EventDispatcher;
	import flash.events.IOErrorEvent;
	import flash.events.ProgressEvent;
	import flash.events.SecurityErrorEvent;
	import flash.net.ObjectEncoding;
	import flash.net.Socket;
	import flash.utils.ByteArray;
	import flash.utils.Endian;
	import flash.utils.IDataInput;
	import flash.utils.IDataOutput;
	import flash.utils.clearTimeout;
	import flash.utils.setTimeout;
	import com.hurlant.crypto.cert.X509Certificate;
	
	
	[Event(name="close", type="flash.events.Event")]
	[Event(name="connect", type="flash.events.Event")]
	[Event(name="ioError", type="flash.events.IOErrorEvent")]
	[Event(name="securityError", type="flash.events.SecurityErrorEvent")]
	[Event(name="socketData", type="flash.events.ProgressEvent")]
	[Event(name="acceptPeerCertificatePrompt", type="flash.events.Event")]
	
	/**
	 * It feels like a socket, but it wraps the stream
	 * over TLS 1.0
	 * 
	 * That's all.
	 * 
	 */
	public class TLSSocket extends Socket implements IDataInput, IDataOutput {
		
		private var _endian:String;
		private var _objectEncoding:uint;
		
		private var _iStream:ByteArray;
		private var _oStream:ByteArray;
		private var _iStream_cursor:uint;
		
		private var _socket:Socket;
		private var _config:TLSConfig;
		private var _engine:TLSEngine;
		public static const ACCEPT_PEER_CERT_PROMPT:String = "acceptPeerCertificatePrompt"
		
		public function TLSSocket(host:String = null, port:int = 0, config:TLSConfig = null) {
			_config = config;
			if (host!=null && port!=0) {
				connect(host, port);
			}
		}
		
		override public function get bytesAvailable():uint {
			return _iStream.bytesAvailable;
		}
		override public function get connected():Boolean {
			return _socket.connected;
		}
		override public function get endian():String {
			return _endian;
		}
		override public function set endian(value:String):void {
			_endian = value;
			_iStream.endian = value;
			_oStream.endian = value;
		}
		override public function get objectEncoding():uint {
			return _objectEncoding;
		}
		override public function set objectEncoding(value:uint):void {
			_objectEncoding = value;
			_iStream.objectEncoding = value;
			_oStream.objectEncoding = value;
		}
		
		
		private function onTLSData(event:TLSEvent):void {
			if (_iStream.position == _iStream.length) {
				_iStream.position = 0;
				_iStream.length = 0;
				_iStream_cursor = 0;
			}
			var cursor:uint = _iStream.position;
			_iStream.position = _iStream_cursor;
			_iStream.writeBytes(event.data);
			_iStream_cursor = _iStream.position;
			_iStream.position = cursor;
			dispatchEvent(new ProgressEvent(ProgressEvent.SOCKET_DATA, false, false, event.data.length));
		}
		
		private function onTLSReady(event:TLSEvent):void {
			_ready = true;
			scheduleWrite();
		}
		
		private function onTLSClose(event:Event):void {
			dispatchEvent(event);
			// trace("Received TLS close");
			close();
		}
		
		private var _ready:Boolean;
		private var _writeScheduler:uint;
		private function scheduleWrite():void {
			if (_writeScheduler!=0) return;
			_writeScheduler = setTimeout(commitWrite, 0);
		}
		private function commitWrite():void {
			clearTimeout(_writeScheduler);
			_writeScheduler = 0;
			if (_ready) {
				_engine.sendApplicationData(_oStream);
				_oStream.length = 0;
			}
		}
		
		
		override public function close():void {
			_ready = false;
			_engine.close();
			if (_socket.connected) {
				_socket.flush();
				_socket.close();
			}
		}
		public function setTLSConfig( config:TLSConfig) : void {
			_config = config;
		}		

		override public function connect(host:String, port:int):void {
			init(new Socket, _config, host);
			_socket.connect(host, port);
			_engine.start();
		}
		
		public function releaseSocket() : void {
			_socket.removeEventListener(Event.CONNECT, dispatchEvent);
			_socket.removeEventListener(IOErrorEvent.IO_ERROR, dispatchEvent);
			_socket.removeEventListener(SecurityErrorEvent.SECURITY_ERROR, dispatchEvent);
			_socket.removeEventListener(Event.CLOSE, dispatchEvent);
			_socket.removeEventListener(ProgressEvent.SOCKET_DATA, _engine.dataAvailable);
			_socket = null; 
		}
		
		public function reinitialize(host:String, config:TLSConfig) : void {
			// Reinitialize the connection using new values
			// but re-use the existing socket
			// Doubt this is useful in any valid context other than my specific case (VMWare)
			var ba:ByteArray = new ByteArray;
			
			if (_socket.bytesAvailable > 0) {
				_socket.readBytes(ba, 0, _socket.bytesAvailable);	
			}
			// Do nothing with it.
			_iStream = new ByteArray;
			_oStream = new ByteArray;
			_iStream_cursor = 0;
			objectEncoding = ObjectEncoding.DEFAULT;
			endian = Endian.BIG_ENDIAN;
			/* 
			_socket.addEventListener(Event.CONNECT, dispatchEvent);
			_socket.addEventListener(IOErrorEvent.IO_ERROR, dispatchEvent);
			_socket.addEventListener(SecurityErrorEvent.SECURITY_ERROR, dispatchEvent);
			_socket.addEventListener(Event.CLOSE, dispatchEvent);
			*/
			
			if (config == null) {
				config = new TLSConfig(TLSEngine.CLIENT);
			}
			
			_engine = new TLSEngine(config, _socket, _socket, host);
			_engine.addEventListener(TLSEvent.DATA, onTLSData);
			_engine.addEventListener(TLSEvent.READY, onTLSReady);
			_engine.addEventListener(Event.CLOSE, onTLSClose);
			_engine.addEventListener(ProgressEvent.SOCKET_DATA, function(e:*):void { _socket.flush(); });
			_socket.addEventListener(ProgressEvent.SOCKET_DATA, _engine.dataAvailable);
			_engine.addEventListener( TLSEvent.PROMPT_ACCEPT_CERT, onAcceptCert );

			_ready = false;
			_engine.start();
		}
		
		public function startTLS(socket:Socket, host:String, config:TLSConfig = null):void {
			if (!socket.connected) {
				throw new Error("Cannot STARTTLS on a socket that isn't connected.");
			}
			init(socket, config, host);
			_engine.start();
		}
		
		private function init(socket:Socket, config:TLSConfig, host:String):void {
			_iStream = new ByteArray;
			_oStream = new ByteArray;
			_iStream_cursor = 0;
			objectEncoding = ObjectEncoding.DEFAULT;
			endian = Endian.BIG_ENDIAN;
			_socket = socket;
			_socket.addEventListener(Event.CONNECT, dispatchEvent);
			_socket.addEventListener(IOErrorEvent.IO_ERROR, dispatchEvent);
			_socket.addEventListener(SecurityErrorEvent.SECURITY_ERROR, dispatchEvent);
			_socket.addEventListener(Event.CLOSE, dispatchEvent);
			
			if (config == null) {
				config = new TLSConfig(TLSEngine.CLIENT);
			}
			_engine = new TLSEngine(config, _socket, _socket, host);
			_engine.addEventListener(TLSEvent.DATA, onTLSData);
			_engine.addEventListener( TLSEvent.PROMPT_ACCEPT_CERT, onAcceptCert );
			_engine.addEventListener(TLSEvent.READY, onTLSReady);
			_engine.addEventListener(Event.CLOSE, onTLSClose);
			_engine.addEventListener(ProgressEvent.SOCKET_DATA, function(e:*):void { if(connected) _socket.flush(); });
			_socket.addEventListener(ProgressEvent.SOCKET_DATA, _engine.dataAvailable);

			_ready = false;
		}
		
		override public function flush():void {
			commitWrite();
			_socket.flush();
		}
		
		override public function readBoolean():Boolean {
			return _iStream.readBoolean();
		}
		
		override public function readByte():int {
			return _iStream.readByte();
		}
		
		override public function readBytes(bytes:ByteArray, offset:uint = 0, length:uint = 0):void {
			return _iStream.readBytes(bytes, offset, length);
		}
		
		override public function readDouble():Number {
			return _iStream.readDouble();
		}
		
		override public function readFloat():Number {
			return _iStream.readFloat();
		}
		
		override public function readInt():int {
			return _iStream.readInt();
		}
		
		override public function readMultiByte(length:uint, charSet:String):String {
			return _iStream.readMultiByte(length, charSet);
		}
		
		override public function readObject():* {
			return _iStream.readObject();
		}
		
		override public function readShort():int {
			return _iStream.readShort();
		}
		
		override public function readUnsignedByte():uint {
			return _iStream.readUnsignedByte();
		}
		
		override public function readUnsignedInt():uint {
			return _iStream.readUnsignedInt();
		}
		
		override public function readUnsignedShort():uint {
			return _iStream.readUnsignedShort();
		}
		
		override public function readUTF():String {
			return _iStream.readUTF();
		}
		
		override public function readUTFBytes(length:uint):String {
			return _iStream.readUTFBytes(length);
		}
		
		override public function writeBoolean(value:Boolean):void {
			_oStream.writeBoolean(value);
			scheduleWrite();
		}
		
		override public function writeByte(value:int):void {
			_oStream.writeByte(value);
			scheduleWrite();
		}
		
		override public function writeBytes(bytes:ByteArray, offset:uint = 0, length:uint = 0):void {
			_oStream.writeBytes(bytes, offset, length);
			scheduleWrite();
		}
		
		override public function writeDouble(value:Number):void {
			_oStream.writeDouble(value);
			scheduleWrite();
		}
		
		override public function writeFloat(value:Number):void {
			_oStream.writeFloat(value);
			scheduleWrite();
		}
		
		override public function writeInt(value:int):void {
			_oStream.writeInt(value);
			scheduleWrite();
		}
		
		override public function writeMultiByte(value:String, charSet:String):void {
			_oStream.writeMultiByte(value, charSet);
			scheduleWrite();
		}
		
		override public function writeObject(object:*):void {
			_oStream.writeObject(object);
			scheduleWrite();
		}
		
		override public function writeShort(value:int):void {
			_oStream.writeShort(value);
			scheduleWrite();
		}
		
		override public function writeUnsignedInt(value:uint):void {
			_oStream.writeUnsignedInt(value);
			scheduleWrite();
		}
		
		override public function writeUTF(value:String):void {
			_oStream.writeUTF(value);
			scheduleWrite();
		}
		
		override public function writeUTFBytes(value:String):void {
			_oStream.writeUTFBytes(value);
			scheduleWrite();
		}
		
		public function getPeerCertificate() : X509Certificate {
			return _engine.peerCertificate;
		}
		
		public function onAcceptCert( event:TLSEvent ) : void {
			dispatchEvent( new TLSSocketEvent( _engine.peerCertificate ) );
		}
		
		// These are just a passthroughs to the engine. Encapsulation, et al
		public function acceptPeerCertificate( event:Event ) : void {
			_engine.acceptPeerCertificate();
		}
	
		public function rejectPeerCertificate( event:Event ) : void {
			_engine.rejectPeerCertificate();
		}
		
	}
}
	
