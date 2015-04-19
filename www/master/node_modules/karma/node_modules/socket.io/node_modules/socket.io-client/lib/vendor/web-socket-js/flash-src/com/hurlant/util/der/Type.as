/**
 * Type
 * 
 * A few Asn-1 structures
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.util.der
{
	import com.hurlant.util.Hex;
	
	public class Type
	{
		public static const TLS_CERT:Array = [ 
			{name:"signedCertificate", extract:true, value: [
				{name:"versionHolder", optional:true, value: [
					{name:"version"}
				], defaultValue: function():Sequence{ 
						var s:Sequence = new Sequence(0, 0); 
						var v:Integer = new Integer(2,1, Hex.toArray("00"));
						s.push(v);
						s.version = v;
						return s;
					}()
				},
				{name:"serialNumber"},
				{name:"signature", value: [
					{name:"algorithmId"}
				]},
				{name:"issuer", extract:true, value: [ 
					{name:"type"},
					{name:"value"}
				]},
				{name:"validity", value: [
					{name:"notBefore"},
					{name:"notAfter"}
				]},
				{name:"subject", extract:true, value: [
				]},
				{name:"subjectPublicKeyInfo", value: [
					{name:"algorithm", value: [
						{name:"algorithmId"}
					]},
					{name:"subjectPublicKey"}
				]},
				{name:"extensions", value: [
				]}
			]}, 
			{name:"algorithmIdentifier",value:[
				{name:"algorithmId"}
			]}, 
			{name:"encrypted", value:null}
		];
		public static const CERTIFICATE:Array = [
			{name:"tbsCertificate", value:[
				{name:"tag0", value:[
					{name:"version"}
				]},
				{name:"serialNumber"},
				{name:"signature"},
				{name:"issuer", value:[
					{name:"type"},
					{name:"value"}
				]},
				{name:"validity", value:[
					{name:"notBefore"},
					{name:"notAfter"}
				]},
				{name:"subject"},
				{name:"subjectPublicKeyInfo", value:[
					{name:"algorithm"},
					{name:"subjectPublicKey"}
				]},
				{name:"issuerUniqueID"},
				{name:"subjectUniqueID"},
				{name:"extensions"}
			]},
			{name:"signatureAlgorithm"},
			{name:"signatureValue"}
		];
		public static const RSA_PUBLIC_KEY:Array = [
			{name:"modulus"},
			{name:"publicExponent"}
		];
		public static const RSA_SIGNATURE:Array = [
			{name:"algorithm", value:[
				{name:"algorithmId"}
			 ]},
			{name:"hash"}
		];
		
	}
}