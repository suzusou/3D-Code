let lastOrientation = null; // 前回の向きを記憶する変数

$(window).on('load orientationchange resize', function() {
    const currentOrientation = Math.abs(window.orientation) === 90 ? 'landscape' : 'portrait';

    if (currentOrientation !== lastOrientation) {
        if (currentOrientation === 'landscape') {
            // 横向きになったときの処理
            alert('横向きになりました');
        } else {
            // 縦向きになったときの処理
            alert('縦向きになりました');
        }
        // 向きを更新
        lastOrientation = currentOrientation;
    }
});