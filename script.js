import {
    FaceLandmarker,
    HandLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs";


// =====================================================
// ELEMENTS
// =====================================================

const video = document.getElementById("video");
const startButton = document.getElementById("startButton");

const actionText = document.getElementById("action");
const confidenceText = document.getElementById("confidence");
const resultImage = document.getElementById("resultImage");


// =====================================================
// MODELS
// =====================================================

let faceLandmarker;
let handLandmarker;

let lastVideoTime = -1;


// =====================================================
// TONGUE DETECTION CANVAS
// =====================================================

const tongueCanvas = document.createElement("canvas");

const tongueCtx =
    tongueCanvas.getContext(
        "2d",
        {
            willReadFrequently: true
        }
    );


// =====================================================
// REFERENCE IMAGES
// =====================================================

const images = {

    eye: "images/eye.png",

    hand: "images/hand.png",

    tongue: "images/tongue.png",

    sad: "images/sad.png",

    angry: "images/angry.png",

    smile: "images/smile.png"

};


// =====================================================
// LOAD MEDIAPIPE MODELS
// =====================================================

async function loadModels() {

    try {

        console.log(
            "Loading MediaPipe models..."
        );


        const vision =
            await FilesetResolver.forVisionTasks(
                "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
            );


        // =============================================
        // FACE LANDMARKER
        // =============================================

        faceLandmarker =
            await FaceLandmarker.createFromOptions(
                vision,
                {

                    baseOptions: {

                        modelAssetPath:
                            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",

                        delegate: "GPU"

                    },

                    outputFaceBlendshapes: true,

                    runningMode: "VIDEO",

                    numFaces: 1

                }
            );


        // =============================================
        // HAND LANDMARKER
        // =============================================

        handLandmarker =
            await HandLandmarker.createFromOptions(
                vision,
                {

                    baseOptions: {

                        modelAssetPath:
                            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",

                        delegate: "GPU"

                    },

                    runningMode: "VIDEO",

                    numHands: 2

                }
            );


        console.log(
            "Models loaded successfully."
        );

    }

    catch (error) {

        console.error(
            "Error loading models:",
            error
        );

    }

}


// =====================================================
// START CAMERA
// =====================================================

startButton.addEventListener(
    "click",
    async () => {

        try {

            const stream =
                await navigator.mediaDevices.getUserMedia({

                    video: {

                        facingMode: "user"

                    },

                    audio: false

                });


            video.srcObject = stream;


            startButton.style.display =
                "none";


            video.addEventListener(
                "loadeddata",
                () => {

                    console.log(
                        "Camera started."
                    );

                    detect();

                },
                {
                    once: true
                }
            );

        }

        catch (error) {

            console.error(
                "Camera error:",
                error
            );

            alert(
                "Camera error: " +
                error.message
            );

        }

    }
);


// =====================================================
// DETECTION LOOP
// =====================================================

function detect() {

    if (
        !faceLandmarker ||
        !handLandmarker
    ) {

        requestAnimationFrame(
            detect
        );

        return;

    }


    if (
        video.readyState >= 2 &&
        video.currentTime !== lastVideoTime
    ) {

        lastVideoTime =
            video.currentTime;


        const timestamp =
            performance.now();


        // =============================================
        // FACE
        // =============================================

        const faceResults =
            faceLandmarker.detectForVideo(
                video,
                timestamp
            );


        // =============================================
        // HAND
        // =============================================

        const handResults =
            handLandmarker.detectForVideo(
                video,
                timestamp
            );


        // =============================================
        // DETECT ACTION
        // =============================================

        detectAction(
            faceResults,
            handResults
        );

    }


    requestAnimationFrame(
        detect
    );

}


// =====================================================
// GET BLENDSHAPE SCORE
// =====================================================

function getScore(
    blendshapes,
    name
) {

    const item =
        blendshapes.find(
            x =>
                x.categoryName === name
        );


    return item
        ? item.score
        : 0;

}


// =====================================================
// ACTUAL TONGUE DETECTION
// =====================================================
//
// This does NOT use MediaPipe's tongueOut score.
//
// It:
// 1. Finds the mouth using Face Landmarks
// 2. Looks at the camera pixels around the mouth
// 3. Searches for pink/red tongue-like pixels
// 4. Checks that the mouth is open
//
// =====================================================

function detectTongue(
    video,
    faceLandmarks,
    jawOpen
) {

    if (
        !faceLandmarks ||
        faceLandmarks.length === 0
    ) {

        return {
            detected: false,
            score: 0
        };

    }


    // ---------------------------------------------
    // Mouth needs to be open
    // ---------------------------------------------

    if (jawOpen < 0.12) {

        return {
            detected: false,
            score: 0
        };

    }


    const width =
        video.videoWidth;

    const height =
        video.videoHeight;


    if (
        !width ||
        !height
    ) {

        return {
            detected: false,
            score: 0
        };

    }


    // ---------------------------------------------
    // Draw current camera frame
    // ---------------------------------------------

    tongueCanvas.width =
        width;

    tongueCanvas.height =
        height;


    tongueCtx.drawImage(
        video,
        0,
        0,
        width,
        height
    );


    // ---------------------------------------------
    // MOUTH LANDMARKS
    // ---------------------------------------------

    const leftCorner =
        faceLandmarks[61];

    const rightCorner =
        faceLandmarks[291];

    const upperLip =
        faceLandmarks[13];

    const lowerLip =
        faceLandmarks[14];


    if (
        !leftCorner ||
        !rightCorner ||
        !upperLip ||
        !lowerLip
    ) {

        return {
            detected: false,
            score: 0
        };

    }


    // ---------------------------------------------
    // MOUTH BOUNDING BOX
    // ---------------------------------------------

    const mouthLeft =
        Math.min(
            leftCorner.x,
            rightCorner.x
        );


    const mouthRight =
        Math.max(
            leftCorner.x,
            rightCorner.x
        );


    const mouthTop =
        Math.min(
            upperLip.y,
            lowerLip.y
        );


    const mouthBottom =
        Math.max(
            upperLip.y,
            lowerLip.y
        );


    const mouthWidth =
        mouthRight -
        mouthLeft;


    const mouthHeight =
        mouthBottom -
        mouthTop;


    if (
        mouthWidth < 0.04 ||
        mouthHeight < 0.015
    ) {

        return {
            detected: false,
            score: 0
        };

    }


    // ---------------------------------------------
    // EXPAND REGION DOWNWARD
    //
    // Tongue can extend below the mouth.
    // ---------------------------------------------

    const regionLeft =
        Math.max(
            0,
            mouthLeft -
            mouthWidth * 0.10
        );


    const regionRight =
        Math.min(
            1,
            mouthRight +
            mouthWidth * 0.10
        );


    const regionTop =
        Math.max(
            0,
            mouthTop
        );


    const regionBottom =
        Math.min(
            1,
            mouthBottom +
            mouthHeight * 5.0
        );


    // ---------------------------------------------
    // Convert to pixels
    // ---------------------------------------------

    const x1 =
        Math.floor(
            regionLeft * width
        );


    const y1 =
        Math.floor(
            regionTop * height
        );


    const x2 =
        Math.floor(
            regionRight * width
        );


    const y2 =
        Math.floor(
            regionBottom * height
        );


    const regionWidth =
        x2 - x1;


    const regionHeight =
        y2 - y1;


    if (
        regionWidth <= 0 ||
        regionHeight <= 0
    ) {

        return {
            detected: false,
            score: 0
        };

    }


    // ---------------------------------------------
    // GET CAMERA PIXELS
    // ---------------------------------------------

    const imageData =
        tongueCtx.getImageData(
            x1,
            y1,
            regionWidth,
            regionHeight
        );


    const pixels =
        imageData.data;


    let tonguePixels = 0;

    let validPixels = 0;


    // ---------------------------------------------
    // CHECK PIXELS
    // ---------------------------------------------

    for (
        let i = 0;
        i < pixels.length;
        i += 4
    ) {

        const r =
            pixels[i];

        const g =
            pixels[i + 1];

        const b =
            pixels[i + 2];


        // Ignore very dark pixels

        if (
            r < 45 &&
            g < 35 &&
            b < 35
        ) {

            continue;

        }


        validPixels++;


        // -----------------------------------------
        // RGB → HSV
        // -----------------------------------------

        const max =
            Math.max(
                r,
                g,
                b
            );


        const min =
            Math.min(
                r,
                g,
                b
            );


        const delta =
            max - min;


        if (
            delta === 0
        ) {

            continue;

        }


        let hue;


        if (
            max === r
        ) {

            hue =
                60 *
                (
                    (
                        (g - b) /
                        delta
                    ) % 6
                );

        }

        else if (
            max === g
        ) {

            hue =
                60 *
                (
                    (
                        (b - r) /
                        delta
                    ) + 2
                );

        }

        else {

            hue =
                60 *
                (
                    (
                        (r - g) /
                        delta
                    ) + 4
                );

        }


        if (
            hue < 0
        ) {

            hue += 360;

        }


        const saturation =
            delta / max;


        const value =
            max / 255;


        // -----------------------------------------
        // TONGUE COLOR
        // -----------------------------------------

        const pinkRed =
            (
                (
                    hue >= 0 &&
                    hue <= 30
                )

                ||

                (
                    hue >= 330 &&
                    hue <= 360
                )
            );


        const sufficientSaturation =
            saturation > 0.20;


        const sufficientBrightness =
            value > 0.25;


        const redDominant =
            r > g * 1.03 &&
            r > b * 1.05;


        if (
            pinkRed &&
            sufficientSaturation &&
            sufficientBrightness &&
            redDominant
        ) {

            tonguePixels++;

        }

    }


    // ---------------------------------------------
    // CALCULATE RATIO
    // ---------------------------------------------

    if (
        validPixels === 0
    ) {

        return {
            detected: false,
            score: 0
        };

    }


    const colorRatio =
        tonguePixels /
        validPixels;


    // ---------------------------------------------
    // DETECTION THRESHOLD
    // ---------------------------------------------

    const detected =
        colorRatio > 0.025;


    const score =
        Math.min(
            colorRatio / 0.12,
            1
        );


    return {
        detected,
        score
    };

}


// =====================================================
// ACTION DETECTION
// =====================================================

function detectAction(
    faceResults,
    handResults
) {


    // =================================================
    // NO FACE
    // =================================================

    if (
        !faceResults.faceBlendshapes ||
        faceResults.faceBlendshapes.length === 0
    ) {

        actionText.textContent =
            "No face detected";

        confidenceText.textContent =
            "Confidence: --";

        return;

    }


    const blendshapes =
        faceResults
            .faceBlendshapes[0]
            .categories;


    // =================================================
    // FACE FEATURES
    // =================================================


    // -----------------------------------------------
    // Pucker
    // -----------------------------------------------

    const pucker =
        getScore(
            blendshapes,
            "mouthPucker"
        );


    // -----------------------------------------------
    // Funnel
    // -----------------------------------------------

    const funnel =
        getScore(
            blendshapes,
            "mouthFunnel"
        );


    // -----------------------------------------------
    // Jaw
    // -----------------------------------------------

    const jawOpen =
        getScore(
            blendshapes,
            "jawOpen"
        );


    // -----------------------------------------------
    // Sad / Frown
    // -----------------------------------------------

    const frownLeft =
        getScore(
            blendshapes,
            "mouthFrownLeft"
        );


    const frownRight =
        getScore(
            blendshapes,
            "mouthFrownRight"
        );


    // -----------------------------------------------
    // Angry
    // -----------------------------------------------

    const mouthPressLeft =
        getScore(
            blendshapes,
            "mouthPressLeft"
        );


    const mouthPressRight =
        getScore(
            blendshapes,
            "mouthPressRight"
        );


    const browDownLeft =
        getScore(
            blendshapes,
            "browDownLeft"
        );


    const browDownRight =
        getScore(
            blendshapes,
            "browDownRight"
        );


    // -----------------------------------------------
    // Smile
    // -----------------------------------------------

    const smileLeft =
        getScore(
            blendshapes,
            "mouthSmileLeft"
        );


    const smileRight =
        getScore(
            blendshapes,
            "mouthSmileRight"
        );


    const smile =
        Math.max(
            smileLeft,
            smileRight
        );


    // =================================================
    // ACTUAL TONGUE DETECTION
    // =================================================

    const tongueResult =
        detectTongue(
            video,
            faceResults.faceLandmarks[0],
            jawOpen
        );


    // =================================================
    // 1. HAND ON CHEEK
    // =================================================

    if (
        handResults.landmarks &&
        handResults.landmarks.length > 0
    ) {

        const hand =
            handResults.landmarks[0];


        const face =
            faceResults.faceLandmarks[0];


        if (
            handNearFace(
                hand,
                face
            )
        ) {

            showResult(
                "Hand on Cheek",
                90,
                images.hand
            );

            return;

        }

    }


    // =================================================
    // 2. TONGUE OUT
    // =================================================

    if (
        tongueResult.detected
    ) {

        showResult(
            "Tongue Out",
            Math.max(
                35,
                tongueResult.score * 100
            ),
            images.tongue
        );

        return;

    }


    // =================================================
    // 3. SMILE
    // =================================================

    const averageSmile =
        (
            smileLeft +
            smileRight
        ) / 2;


    if (
        averageSmile > 0.40 &&
        smileLeft > 0.32 &&
        smileRight > 0.32 &&
        jawOpen < 0.35 &&
        !tongueResult.detected
    ) {

        const confidence =
            averageSmile;


        showResult(
            "Smile",
            confidence * 100,
            images.smile
        );

        return;

    }


    // =================================================
    // 4. EYE / PUCKER
    // =================================================

    if (
        pucker > 0.45 ||
        funnel > 0.45
    ) {

        const confidence =
            Math.max(
                pucker,
                funnel
            );


        showResult(
            "Eye / Pucker",
            confidence * 100,
            images.eye
        );

        return;

    }


    // =================================================
    // 5. ANGRY
    // =================================================

    if (

        browDownLeft > 0.40 &&

        browDownRight > 0.40 &&

        (

            mouthPressLeft > 0.25 ||

            mouthPressRight > 0.25

        )

    ) {

        const confidence =
            Math.max(

                browDownLeft,

                browDownRight,

                mouthPressLeft,

                mouthPressRight

            );


        showResult(
            "Angry",
            confidence * 100,
            images.angry
        );

        return;

    }


    // =================================================
    // 6. SAD / POUT
    // =================================================

    if (

        frownLeft > 0.35 ||

        frownRight > 0.35 ||

        (

            browDownLeft > 0.50 &&

            browDownRight > 0.50

        )

    ) {

        const confidence =
            Math.max(

                frownLeft,

                frownRight,

                browDownLeft,

                browDownRight

            );


        showResult(
            "Sad / Pout",
            confidence * 100,
            images.sad
        );

        return;

    }


    // =================================================
    // NO ACTION
    // =================================================

    actionText.textContent =
        "No action";


    confidenceText.textContent =
        "Confidence: --";

}


// =====================================================
// HAND NEAR FACE
// =====================================================

function handNearFace(
    hand,
    face
) {

    const nose =
        face[1];


    const wrist =
        hand[0];


    const distance =
        Math.sqrt(

            Math.pow(
                wrist.x - nose.x,
                2
            )

            +

            Math.pow(
                wrist.y - nose.y,
                2
            )

        );


    return distance < 0.40;

}


// =====================================================
// SHOW RESULT
// =====================================================

function showResult(
    name,
    confidence,
    image
) {

    actionText.textContent =
        name;


    confidenceText.textContent =
        "Confidence: " +
        Math.round(confidence) +
        "%";


    if (image) {

        resultImage.src =
            image;

    }

}


// =====================================================
// START MODEL
// =====================================================

loadModels();