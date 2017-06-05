// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package testtext contains test data that is of common use to the text
// repository.
package testtext // import "golang.org/x/text/internal/testtext"

const (

	// ASCII is an ASCII string containing all letters in the English alphabet.
	ASCII = "The quick brown fox jumps over the lazy dog. " +
		"The quick brown fox jumps over the lazy dog. " +
		"The quick brown fox jumps over the lazy dog. " +
		"The quick brown fox jumps over the lazy dog. " +
		"The quick brown fox jumps over the lazy dog. " +
		"The quick brown fox jumps over the lazy dog. " +
		"The quick brown fox jumps over the lazy dog. " +
		"The quick brown fox jumps over the lazy dog. " +
		"The quick brown fox jumps over the lazy dog. " +
		"The quick brown fox jumps over the lazy dog. "

	// Vietnamese is a snippet from http://creativecommons.org/licenses/by-sa/3.0/vn/
	Vietnamese = `Với các điều kiện sau: Ghi nhận công của tác giả. 
Nếu bạn sử dụng, chuyển đổi, hoặc xây dựng dự án từ 
nội dung được chia sẻ này, bạn phải áp dụng giấy phép này hoặc 
một giấy phép khác có các điều khoản tương tự như giấy phép này
cho dự án của bạn. Hiểu rằng: Miễn — Bất kỳ các điều kiện nào
trên đây cũng có thể được miễn bỏ nếu bạn được sự cho phép của
người sở hữu bản quyền. Phạm vi công chúng — Khi tác phẩm hoặc
bất kỳ chương nào của tác phẩm đã trong vùng dành cho công
chúng theo quy định của pháp luật thì tình trạng của nó không 
bị ảnh hưởng bởi giấy phép trong bất kỳ trường hợp nào.`

	// Russian is a snippet from http://creativecommons.org/licenses/by-sa/1.0/deed.ru
	Russian = `При обязательном соблюдении следующих условий:
Attribution — Вы должны атрибутировать произведение (указывать
автора и источник) в порядке, предусмотренном автором или
лицензиаром (но только так, чтобы никоим образом не подразумевалось,
что они поддерживают вас или использование вами данного произведения).
Υπό τις ακόλουθες προϋποθέσεις:`

	// Greek is a snippet from http://creativecommons.org/licenses/by-sa/3.0/gr/
	Greek = `Αναφορά Δημιουργού — Θα πρέπει να κάνετε την αναφορά στο έργο με τον
τρόπο που έχει οριστεί από το δημιουργό ή το χορηγούντο την άδεια
(χωρίς όμως να εννοείται με οποιονδήποτε τρόπο ότι εγκρίνουν εσάς ή
τη χρήση του έργου από εσάς). Παρόμοια Διανομή — Εάν αλλοιώσετε,
τροποποιήσετε ή δημιουργήσετε περαιτέρω βασισμένοι στο έργο θα
μπορείτε να διανέμετε το έργο που θα προκύψει μόνο με την ίδια ή
παρόμοια άδεια.`

	// Arabic is a snippet from http://creativecommons.org/licenses/by-sa/3.0/deed.ar
	Arabic = `بموجب الشروط التالية نسب المصنف — يجب عليك أن
تنسب العمل بالطريقة التي تحددها المؤلف أو المرخص (ولكن ليس بأي حال من
الأحوال أن توحي وتقترح بتحول أو استخدامك للعمل).
المشاركة على قدم المساواة — إذا كنت يعدل ، والتغيير ، أو الاستفادة
من هذا العمل ، قد ينتج عن توزيع العمل إلا في ظل تشابه او تطابق فى واحد
لهذا الترخيص.`

	// Hebrew is a snippet from http://creativecommons.org/licenses/by-sa/1.0/il/
	Hebrew = `בכפוף לתנאים הבאים: ייחוס — עליך לייחס את היצירה (לתת קרדיט) באופן
המצויין על-ידי היוצר או מעניק הרישיון (אך לא בשום אופן המרמז על כך
שהם תומכים בך או בשימוש שלך ביצירה). שיתוף זהה — אם תחליט/י לשנות,
לעבד או ליצור יצירה נגזרת בהסתמך על יצירה זו, תוכל/י להפיץ את יצירתך
החדשה רק תחת אותו הרישיון או רישיון דומה לרישיון זה.`

	TwoByteUTF8 = Russian + Greek + Arabic + Hebrew

	// Thai is a snippet from http://creativecommons.org/licenses/by-sa/3.0/th/
	Thai = `ภายใต้เงื่อนไข ดังต่อไปนี้ : แสดงที่มา — คุณต้องแสดงที่
มาของงานดังกล่าว ตามรูปแบบที่ผู้สร้างสรรค์หรือผู้อนุญาตกำหนด (แต่
ไม่ใช่ในลักษณะที่ว่า พวกเขาสนับสนุนคุณหรือสนับสนุนการที่
คุณนำงานไปใช้) อนุญาตแบบเดียวกัน — หากคุณดัดแปลง เปลี่ยนรูป หรื
อต่อเติมงานนี้ คุณต้องใช้สัญญาอนุญาตแบบเดียวกันหรือแบบที่เหมื
อนกับสัญญาอนุญาตที่ใช้กับงานนี้เท่านั้น`

	ThreeByteUTF8 = Thai

	// Japanese is a snippet from http://creativecommons.org/licenses/by-sa/2.0/jp/
	Japanese = `あなたの従うべき条件は以下の通りです。
表示 — あなたは原著作者のクレジットを表示しなければなりません。
継承 — もしあなたがこの作品を改変、変形または加工した場合、
あなたはその結果生じた作品をこの作品と同一の許諾条件の下でのみ
頒布することができます。`

	// Chinese is a snippet from http://creativecommons.org/licenses/by-sa/2.5/cn/
	Chinese = `您可以自由： 复制、发行、展览、表演、放映、
广播或通过信息网络传播本作品 创作演绎作品
对本作品进行商业性使用 惟须遵守下列条件：
署名 — 您必须按照作者或者许可人指定的方式对作品进行署名。
相同方式共享 — 如果您改变、转换本作品或者以本作品为基础进行创作，
您只能采用与本协议相同的许可协议发布基于本作品的演绎作品。`

	// Korean is a snippet from http://creativecommons.org/licenses/by-sa/2.0/kr/
	Korean = `다음과 같은 조건을 따라야 합니다: 저작자표시
— 저작자나 이용허락자가 정한 방법으로 저작물의
원저작자를 표시하여야 합니다(그러나 원저작자가 이용자나 이용자의
이용을 보증하거나 추천한다는 의미로 표시해서는 안됩니다). 
동일조건변경허락 — 이 저작물을 이용하여 만든 이차적 저작물에는 본
라이선스와 동일한 라이선스를 적용해야 합니다.`

	CJK = Chinese + Japanese + Korean

	All = ASCII + Vietnamese + TwoByteUTF8 + ThreeByteUTF8 + CJK
)
