/**
 * X509CertificateCollection
 * 
 * A class to store and index X509 Certificates by Subject.
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.cert {
	
	public class X509CertificateCollection {
		
		private var _map:Object;
		
		public function X509CertificateCollection() {
			_map = {};
		}
		
		/**
		 * Mostly meant for built-in CA loading.
		 * This entry-point allows to index CAs without parsing them.
		 * 
		 * @param name		A friendly name. not currently used
		 * @param subject	base64 DER encoded Subject principal for the Cert
		 * @param pem		PEM encoded certificate data
		 * 
		 */
		public function addPEMCertificate(name:String, subject:String, pem:String):void {
			_map[subject] = new X509Certificate(pem);
		}
		
		/**
		 * Adds a X509 certificate to the collection.
		 * This call will force the certificate to be parsed.
		 * 
		 * @param cert		A X509 certificate
		 * 
		 */
		public function addCertificate(cert:X509Certificate):void {
			var subject:String = cert.getSubjectPrincipal();
			_map[subject] = cert;
		}
		
		/**
		 * Returns a X509 Certificate present in the collection, given
		 * a base64 DER encoded X500 Subject principal
		 * 
		 * @param subject	A Base64 DER-encoded Subject principal
		 * @return 			A matching certificate, or null.
		 * 
		 */
		public function getCertificate(subject:String):X509Certificate {
			return _map[subject];
		}
		
	}
}