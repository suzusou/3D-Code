// カメラとボタンの要素を取得
const cameraElement = document.getElementById("camera");
const takeButton = document.getElementById("btn_camera");

const text = document.getElementById("text_result");

// カメラ映像を表示するためのvideo要素を作成
const video = document.createElement("video");
video.setAttribute("autoplay", true);
video.setAttribute("playsinline", true); // iOSの互換性のため

// 結果を表示する用のテキスト
const loadtext = document.getElementById("loader-text");

// カメラ部分にvideo要素を追加
cameraElement.appendChild(video);


// 外側カメラをデフォルトで選択
async function startCamera() {
    // stopCamera();
    try {
        const constraints = {
            video: {
                facingMode: { exact: "environment" }, // 外カメラを指定
            },
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
    } catch (error) {
        console.error("カメラのアクセスに失敗しました:", error);

        // 外カメラが利用できない場合は内カメラにフォールバック
        if (
            error.name === "OverconstrainedError" ||
            error.name === "ConstraintNotSatisfiedError"
        ) {
            console.log("外カメラが利用できないため、内カメラを使用します。");
            const fallbackStream = await navigator.mediaDevices.getUserMedia({
                video: true,
            });
            video.srcObject = fallbackStream;
        }
    }
}


// 写真を撮って送信する
takeButton.addEventListener("click", async () => {

    takeButton.innerHTML = `<div class="loading"></div>`;
    // setTimeout(() => {
    //     takeButton.innerHTML = "送信済み";
    // }, 10000)

    loadtext.style.display = 'block';
    
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");

    // canvasのサイズをvideoのサイズに合わせる
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // canvasに現在の映像を描画
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Base64データを取得
    const base64_data = canvas.toDataURL("image/png");

    // stopCamera();
    // コンソールにBase64データを表示
    // console.log("Base64 Image:", base64_data);

    // FastAPIに送信
    try {
        const response = await fetch("https://yax.f5.si/api/upload_image", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ image_data: base64_data.split(",")[1] }), // "data:image/png;base64,"を除去
        });
        const result = await response.json();
        const regex = /^https:\/\//;


        console.log(result);
        alert(result.message);
        takeButton.innerHTML = "Take";
        text.style.display = "block";

        if(regex.test(result.message)){
            text.innerHTML = `<a href = "${result.message}">${result.message}</a>`
        }else{
            text.textContent = result.message;
        }
        

        
        const agent = navigator.userAgent.toLowerCase();

        //ios端末の検知
        //標準だと、ios端末(iPhone,iPad)だとカメラ撮影後フリーズしてしまうため、カメラを再起動
        if(navigator.userAgent.match(/iPhone|iPad|macintosh/)){
            startCamera();
            //window.location.reload()
        } else if(/ipad|macintosh/.test(agent) && ('ontouchend' in document)){
            startCamera();
        } else {
            // console.log("iPhoneではないです")
            // text.textContent = "iPhoneではない"
        }
    } catch (error) {
        console.error("エラーが発生しました:", error);
        alert("送信に失敗しました");
        takeButton.innerHTML = "Take";
        text.style.display = "block";
        text.textContent = "画像を読み取れませんでした";
    }

    // 読み取りボタンが押され、デコード処理を終えたテキストを表示
    // let addText =  "result load";
    // let text = document.getElementById("text_result");
    // text.style.display = "block";
    // text.textContent = addText;

});



// カメラの起動
startCamera();



