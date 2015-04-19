/**
 * BlowFishKeyTest
 * 
 * A test class for BlowFishKey
 * Copyright (c) 2007 Henri Torgemane
 * 
 * See LICENSE.txt for full license information.
 */
package com.hurlant.crypto.tests
{
	import com.hurlant.crypto.symmetric.BlowFishKey;
	import com.hurlant.util.Hex;
	import flash.utils.ByteArray;
	
	public class BlowFishKeyTest extends TestCase
	{
		public function BlowFishKeyTest(h:ITestHarness)
		{
			super(h, "BlowFishKey Test");
			runTest(testECB,"BlowFish ECB Test Vectors");
			h.endTestCase();
		}
		
		/**
		 * Test vectors from http://www.schneier.com/code/vectors.txt
		 */
		public function testECB():void {
			var keys:Array = [
			"0000000000000000",
			"FFFFFFFFFFFFFFFF",
			"3000000000000000",
			"1111111111111111",
			"0123456789ABCDEF",
			"1111111111111111",
			"0000000000000000",
			"FEDCBA9876543210",
			"7CA110454A1A6E57",
			"0131D9619DC1376E",
			"07A1133E4A0B2686",
			"3849674C2602319E",
			"04B915BA43FEB5B6",
			"0113B970FD34F2CE",
			"0170F175468FB5E6",
			"43297FAD38E373FE",
			"07A7137045DA2A16",
			"04689104C2FD3B2F",
			"37D06BB516CB7546",
			"1F08260D1AC2465E",
			"584023641ABA6176",
			"025816164629B007",
			"49793EBC79B3258F",
			"4FB05E1515AB73A7",
			"49E95D6D4CA229BF",
			"018310DC409B26D6",
			"1C587F1C13924FEF",
			"0101010101010101",
			"1F1F1F1F0E0E0E0E",
			"E0FEE0FEF1FEF1FE",
			"0000000000000000",
			"FFFFFFFFFFFFFFFF",
			"0123456789ABCDEF",
			"FEDCBA9876543210" ];
			var pts:Array = [
			"0000000000000000",
			"FFFFFFFFFFFFFFFF",
			"1000000000000001",
			"1111111111111111",
			"1111111111111111",
			"0123456789ABCDEF",
			"0000000000000000",
			"0123456789ABCDEF",
			"01A1D6D039776742",
			"5CD54CA83DEF57DA",
			"0248D43806F67172",
			"51454B582DDF440A",
			"42FD443059577FA2",
			"059B5E0851CF143A",
			"0756D8E0774761D2",
			"762514B829BF486A",
			"3BDD119049372802",
			"26955F6835AF609A",
			"164D5E404F275232",
			"6B056E18759F5CCA",
			"004BD6EF09176062",
			"480D39006EE762F2",
			"437540C8698F3CFA",
			"072D43A077075292",
			"02FE55778117F12A",
			"1D9D5C5018F728C2",
			"305532286D6F295A",
			"0123456789ABCDEF",
			"0123456789ABCDEF",
			"0123456789ABCDEF",
			"FFFFFFFFFFFFFFFF",
			"0000000000000000",
			"0000000000000000",
			"FFFFFFFFFFFFFFFF" ];
			var cts:Array = [
			"4EF997456198DD78",
			"51866FD5B85ECB8A",
			"7D856F9A613063F2",
			"2466DD878B963C9D",
			"61F9C3802281B096",
			"7D0CC630AFDA1EC7",
			"4EF997456198DD78",
			"0ACEAB0FC6A0A28D",
			"59C68245EB05282B",
			"B1B8CC0B250F09A0",
			"1730E5778BEA1DA4",
			"A25E7856CF2651EB",
			"353882B109CE8F1A",
			"48F4D0884C379918",
			"432193B78951FC98",
			"13F04154D69D1AE5",
			"2EEDDA93FFD39C79",
			"D887E0393C2DA6E3",
			"5F99D04F5B163969",
			"4A057A3B24D3977B",
			"452031C1E4FADA8E",
			"7555AE39F59B87BD",
			"53C55F9CB49FC019",
			"7A8E7BFA937E89A3",
			"CF9C5D7A4986ADB5",
			"D1ABB290658BC778",
			"55CB3774D13EF201",
			"FA34EC4847B268B2",
			"A790795108EA3CAE",
			"C39E072D9FAC631D",
			"014933E0CDAFF6E4",
			"F21E9A77B71C49BC",
			"245946885754369A",
			"6B5C5A9C5D9E0A5A" ];

			for (var i:uint=0;i<keys.length;i++) {
				var key:ByteArray = Hex.toArray(keys[i]);
				var pt:ByteArray = Hex.toArray(pts[i]);
				var bf:BlowFishKey = new BlowFishKey(key);
				bf.encrypt(pt);
				var out:String = Hex.fromArray(pt).toUpperCase();
				assert("comparing "+cts[i]+" to "+out, cts[i]==out);
				// now go back to plaintext
				bf.decrypt(pt);
				out = Hex.fromArray(pt).toUpperCase();
				assert("comparing "+pts[i]+" to "+out, pts[i]==out);
			}
		}
	}
}