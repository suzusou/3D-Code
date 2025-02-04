// // ボタンを押したとき、FastAPIにリクエストを開始
// document.getElementById("btn_generate").addEventListener("click", async () => {
//     try {

//         // FastAPIのエンドポイント(/api/dataは、pythonで指定している)にリクエスト
//         const response = await fetch("http://127.0.0.1:8000/api/data");

//         // レスポンスが成功しなかった場合
//         if(!response.ok) {
//             throw new Error(`HTTPエラー! ステータス: ${response.status}`);
//         }
//         // FastAPIから返されるJSONデータの取得
//         const output = await response.json();

//         // 取得したJSONデータの["message"]のデータを表示
//         // document.getElementById("message").textContent = output["message"]
//         console.log('Image Data: ', output["img"]);

//         // 取得した["img"]にあるBase64データを画像として表示
//         const imgElement = document.createElement('img');
//         imgElement.src = `data:image/png;base64,${output["img"]}`;

//         // 以前に表示されていた画像を削除
//         document.getElementById("image").innerHTML = ''; 
//         document.getElementById("image").appendChild(imgElement);

//     } catch(error){
//         console.log("エラーが発生しました");
//     }
// })

document.getElementById("btn_generate").addEventListener("click", async () => {
    try {
        // 入力フィールドからデータを取得
        const inputField = document.querySelector(".input-field");
        const inputText = inputField.value;

        if (!inputText) {
            alert("入力フィールドが空です");
            return;
        }

        // FastAPIのエンドポイントにPOSTリクエストを送信
        const response = await fetch("https://yax.f5.si/api/data", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: inputText })
        });

        // レスポンスの処理
        if (!response.ok) {
            throw new Error(`HTTPエラー! ステータス: ${response.status}`);
        }

        const output = await response.json();
        console.log('Image Data: ', output["img"]);

        // Base64データを画像として表示
        const imgElement = document.createElement('img');
        imgElement.src = `data:image/png;base64,${output["img"]}`;

        document.getElementById("image").innerHTML = ''; 
        document.getElementById("image").appendChild(imgElement);

    } catch (error) {
        console.error("エラーが発生しました", error);
    }
});

// 生成したコードの画像をダウンロードする処理
document.getElementById("btn_download").addEventListener("click", async () => {
    const imgElement = document.querySelector("#image img");
    // console.log(imgElement);
    // 画像がない場合は、動かないようにする
    if(!imgElement) {
        alert("画像を生成してください");
        return;
    }
    
    // Base64データを取得
    const base64Data = imgElement.src;

    // aタグを生成してダウンロード処理を開始
    const link = document.createElement("a");
    link.href = base64Data;
    link.download = "generate_code.png"
    link.click();
});
