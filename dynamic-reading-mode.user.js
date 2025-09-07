// ==UserScript==
// @name         动态阅读模式
// @namespace    http://tampermonkey.net/
// @version      2.1.0
// @description  Intelligent reading assistant based on face distance detection Full version, including dynamic fonts, contrast adjustments, and a Kindle-style reading experience
// @author       Reading Mode Team
// @match        *://*/*
// @grant        none
// @require      https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js
// @require      https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js
// @require      https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js
// @run-at       document-idle
// ==/UserScript==

(function () {
  "use strict";

  // =====  =====
  class MediaPipeDistanceDetector {
    constructor(config = {}) {
      this.config = {
        onDistanceUpdate: config.onDistanceUpdate || (() => {}),
        onError: config.onError || ((error) => console.error(error)),
        onCalibrationFrameStatus: config.onCalibrationFrameStatus || (() => {}),
        smoothingWindow: config.smoothingWindow || 5,
        minConfidence: config.minConfidence || 0.5,
        distanceScale: config.distanceScale || 100,
        disableCamera: config.disableCamera || false,
        basePath:
          config.basePath ||
          "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/",
        calibrationFrame: {
          width: 200,
          height: 280,
          targetFaceRatio: 0.5,
          positionTolerance: 0.2,
        },
      };

      this.faceMesh = null;
      this.isInitialized = false;
      this.isRunning = false;
      this.video = null;
      this.canvas = null;
      this.canvasCtx = null;
      this.detectionInterval = null;

      this.faceHistory = [];
      this.maxHistorySize = this.config.smoothingWindow;

      this.calibration = {
        isCalibrated: false,
        referenceFaceWidth: null,
        referenceDistance: 0,
        timestamp: null,
      };

      this.lastCalibrationStatus = null;
      this.statusThrottleTimer = null;
      this.statusThrottleDelay = 100;
      this.statusStabilityCounter = 0;
      this.statusStabilityThreshold = 3;

      console.log("📏 MediaPipe Distance Detector 初始化");
    }

    async initialize() {
      if (this.isInitialized) {
        console.log("✅ MediaPipe已初始化");
        return true;
      }

      try {
        console.log("🚀 初始化MediaPipe Face Mesh...");

        this.faceMesh = new FaceMesh({
          locateFile: (file) => {
            return `${this.config.basePath}${file}`;
          },
        });

        this.faceMesh.setOptions({
          maxNumFaces: 1,
          refineLandmarks: false,
          minDetectionConfidence: this.config.minConfidence,
          minTrackingConfidence: this.config.minConfidence,
        });

        this.faceMesh.onResults((results) =>
          this.processFaceMeshResults(results)
        );

        this.isInitialized = true;
        console.log("✅ MediaPipe Face Mesh 初始化成功");
        return true;
      } catch (error) {
        console.error("❌ MediaPipe初始化失败:", error);
        this.config.onError(error);
        return false;
      }
    }

    async startCamera() {
      if (this.config.disableCamera) {
        console.log("⚠️ 相机操作被禁用");
        return false;
      }

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        const error = new Error(
          "您的浏览器不支持访问摄像头，请使用最新版Chrome/Edge/Firefox浏览器"
        );
        console.error("❌", error.message);
        this.config.onError(error);
        return false;
      }

      try {
        this.video = document.createElement("video");
        this.video.style.display = "none";
        this.video.style.width = "1px";
        this.video.style.height = "1px";
        this.video.style.position = "fixed";
        this.video.style.top = "0";
        this.video.style.left = "0";
        this.video.style.zIndex = "-1";
        this.video.playsInline = true;
        this.video.autoplay = true;
        this.video.muted = true;
        document.body.appendChild(this.video);

        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: 640,
            height: 480,
            facingMode: "user",
          },
          audio: false,
        });

        this.video.srcObject = stream;
        await this.video.play();

        console.log("✅ 相机已启动");
        return true;
      } catch (error) {
        console.error("❌ 启动相机失败:", error.message);

        let errorMessage = "无法访问摄像头。";
        if (error.name === "NotAllowedError") {
          errorMessage =
            "您拒绝了摄像头访问权限，请在浏览器设置中允许访问摄像头。";
        } else if (error.name === "NotFoundError") {
          errorMessage =
            "未检测到摄像头设备，请确保您的设备有摄像头并正常连接。";
        }

        const friendlyError = new Error(errorMessage);
        this.config.onError(friendlyError);
        return false;
      }
    }

    stopCamera() {
      if (this.video && this.video.srcObject) {
        const tracks = this.video.srcObject.getTracks();
        tracks.forEach((track) => track.stop());
        this.video.srcObject = null;

        if (this.video.parentNode) {
          this.video.parentNode.removeChild(this.video);
        }
        this.video = null;
      }
    }

    startDetectionLoop() {
      if (!this.isInitialized || !this.video) {
        console.error("❌ MediaPipe或视频未初始化，无法开始检测");
        return false;
      }

      this.isRunning = true;

      const camera = new Camera(this.video, {
        onFrame: async () => {
          if (this.isRunning && this.video) {
            await this.faceMesh.send({ image: this.video });
          }
        },
      });
      camera.start();

      return true;
    }

    processFaceMeshResults(results) {
      if (!this.isRunning) return;

      try {
        if (
          !results ||
          !results.multiFaceLandmarks ||
          results.multiFaceLandmarks.length === 0
        ) {
          this.handleNoFaceDetected();
          return;
        }

        const landmarks = results.multiFaceLandmarks[0];
        const faceMetrics = this.calculateFaceMetrics(
          landmarks,
          results.image.width,
          results.image.height
        );

        this.addToFaceHistory(faceMetrics);
        const smoothedMetrics = this.getSmoothedFaceMetrics();

        const alignmentStatus = this.checkCalibrationFrameAlignment(
          smoothedMetrics,
          results.image.width,
          results.image.height
        );

        let distanceData = this.calculateRelativeDistance(smoothedMetrics);
        distanceData = {
          ...distanceData,
          alignmentStatus,
          faceDetected: true,
        };

        this.config.onDistanceUpdate(distanceData);
      } catch (error) {
        console.error("❌ 处理Face Mesh结果时出错:", error);
        this.config.onError(error);
      }
    }

    calculateFaceMetrics(landmarks, imageWidth, imageHeight) {
      const leftEar = landmarks[234];
      const rightEar = landmarks[454];
      const nose = landmarks[4];
      const forehead = landmarks[10];
      const chin = landmarks[152];

      const faceWidth = Math.abs((rightEar.x - leftEar.x) * imageWidth);
      const faceHeight = Math.abs((chin.y - forehead.y) * imageHeight);

      const faceCenterX = (rightEar.x + leftEar.x) / 2;
      const faceCenterY = (chin.y + forehead.y) / 2;

      const normalizedX = (faceCenterX - 0.5) * 2;
      const normalizedY = (faceCenterY - 0.5) * 2;

      return {
        faceWidth,
        faceHeight,
        faceWidthToImageRatio: faceWidth / imageWidth,
        faceHeightToImageRatio: faceHeight / imageHeight,
        faceCenterX: normalizedX,
        faceCenterY: normalizedY,
        imageWidth,
        imageHeight,
        timestamp: Date.now(),
      };
    }

    addToFaceHistory(metrics) {
      this.faceHistory.push(metrics);
      while (this.faceHistory.length > this.maxHistorySize) {
        this.faceHistory.shift();
      }
    }

    getSmoothedFaceMetrics() {
      if (this.faceHistory.length === 0) return null;

      const count = this.faceHistory.length;
      const smoothed = {
        faceWidth: 0,
        faceHeight: 0,
        faceWidthToImageRatio: 0,
        faceHeightToImageRatio: 0,
        faceCenterX: 0,
        faceCenterY: 0,
        imageWidth: this.faceHistory[count - 1].imageWidth,
        imageHeight: this.faceHistory[count - 1].imageHeight,
        timestamp: this.faceHistory[count - 1].timestamp,
      };

      for (const metrics of this.faceHistory) {
        smoothed.faceWidth += metrics.faceWidth;
        smoothed.faceHeight += metrics.faceHeight;
        smoothed.faceWidthToImageRatio += metrics.faceWidthToImageRatio;
        smoothed.faceHeightToImageRatio += metrics.faceHeightToImageRatio;
        smoothed.faceCenterX += metrics.faceCenterX;
        smoothed.faceCenterY += metrics.faceCenterY;
      }

      smoothed.faceWidth /= count;
      smoothed.faceHeight /= count;
      smoothed.faceWidthToImageRatio /= count;
      smoothed.faceHeightToImageRatio /= count;
      smoothed.faceCenterX /= count;
      smoothed.faceCenterY /= count;

      return smoothed;
    }

    checkCalibrationFrameAlignment(faceMetrics, imageWidth, imageHeight) {
      if (!faceMetrics) {
        console.log("🎯 [CALIBRATION] No face metrics available");
        return "no-face";
      }

      console.log("🎯 [CALIBRATION] Processing face metrics:", {
        faceCenterX: faceMetrics.faceCenterX.toFixed(3),
        faceCenterY: faceMetrics.faceCenterY.toFixed(3),
        faceWidthRatio: faceMetrics.faceWidthToImageRatio.toFixed(3),
      });

      const { targetFaceRatio, positionTolerance } =
        this.config.calibrationFrame;
      const idealFaceRatio = targetFaceRatio;
      const currentFaceRatio = faceMetrics.faceWidthToImageRatio;
      const posTolerance = positionTolerance;

      const isTooLeft = faceMetrics.faceCenterX < -posTolerance;
      const isTooRight = faceMetrics.faceCenterX > posTolerance;
      const isTooHigh = faceMetrics.faceCenterY < -posTolerance;
      const isTooLow = faceMetrics.faceCenterY > posTolerance;

      const isTooFar = currentFaceRatio < idealFaceRatio - idealFaceRatio * 0.5;
      const isTooClose =
        currentFaceRatio > idealFaceRatio + idealFaceRatio * 0.5;

      let status = "good";

      if (isTooFar) {
        status = "too-far";
      } else if (isTooClose) {
        status = "too-close";
      } else if (isTooLeft) {
        status = "too-left";
      } else if (isTooRight) {
        status = "too-right";
      } else if (isTooHigh) {
        status = "too-high";
      } else if (isTooLow) {
        status = "too-low";
      }

      console.log("🎯 [CALIBRATION] Status decision:", status, {
        isTooFar,
        isTooClose,
        isTooLeft,
        isTooRight,
        isTooHigh,
        isTooLow,
        currentFaceRatio: currentFaceRatio.toFixed(3),
        idealFaceRatio: idealFaceRatio.toFixed(3),
      });

      if (this.lastCalibrationStatus !== status) {
        this.statusStabilityCounter = 0;
        this.lastCalibrationStatus = status;
        console.log(
          "🎯 [CALIBRATION] Status changed to:",
          status,
          "resetting counter"
        );
      } else {
        this.statusStabilityCounter++;
        console.log(
          "🎯 [CALIBRATION] Status stable:",
          status,
          "counter:",
          this.statusStabilityCounter,
          "threshold:",
          this.statusStabilityThreshold
        );

        if (this.statusStabilityCounter >= this.statusStabilityThreshold) {
          console.log(
            "🎯 [CALIBRATION] Triggering onCalibrationFrameStatus callback"
          );
          this.config.onCalibrationFrameStatus({
            status,
            faceRatio: currentFaceRatio,
            idealRatio: idealFaceRatio,
            position: {
              x: faceMetrics.faceCenterX,
              y: faceMetrics.faceCenterY,
            },
          });
        }
      }

      return status;
    }

    async calibrate(fontSize = 16) {
      try {
        const metrics = this.getSmoothedFaceMetrics();

        if (!metrics) {
          throw new Error("未检测到人脸，无法校准");
        }

        const alignmentStatus = this.checkCalibrationFrameAlignment(
          metrics,
          metrics.imageWidth,
          metrics.imageHeight
        );

        if (alignmentStatus !== "good") {
          let errorMessage = "请调整位置和距离";

          switch (alignmentStatus) {
            case "too-far":
              errorMessage = "请靠近一点";
              break;
            case "too-close":
              errorMessage = "请远离一点";
              break;
            case "too-left":
              errorMessage = "请向右移动";
              break;
            case "too-right":
              errorMessage = "请向左移动";
              break;
            case "too-high":
              errorMessage = "请向下移动";
              break;
            case "too-low":
              errorMessage = "请向上移动";
              break;
          }

          throw new Error(errorMessage);
        }

        this.calibration = {
          isCalibrated: true,
          referenceFaceWidth: metrics.faceWidth,
          referenceDistance: 0,
          referenceFontSize: fontSize,
          timestamp: Date.now(),
        };

        console.log("✅ 校准成功:", this.calibration);

        localStorage.setItem(
          "mediapipe-calibration",
          JSON.stringify(this.calibration)
        );

        return this.calibration;
      } catch (error) {
        console.error("❌ 校准失败:", error);
        throw error;
      }
    }

    calculateRelativeDistance(faceMetrics) {
      if (!this.calibration.isCalibrated || !faceMetrics) {
        return {
          distance: 0,
          offset: 0,
          distanceRatio: 1,
          faceWidth: faceMetrics ? faceMetrics.faceWidth : 0,
          isCalibrated: false,
        };
      }

      const ratio = faceMetrics.faceWidth / this.calibration.referenceFaceWidth;
      let relativeDistance = (1 / ratio - 1) * this.config.distanceScale;

      return {
        distance: relativeDistance,
        offset: relativeDistance,
        distanceRatio: ratio,
        faceWidth: faceMetrics.faceWidth,
        isCalibrated: true,
      };
    }

    handleNoFaceDetected() {
      const lastDetectedFace =
        this.faceHistory.length > 0
          ? this.faceHistory[this.faceHistory.length - 1].timestamp
          : 0;

      if (Date.now() - lastDetectedFace > 500) {
        this.faceHistory = [];

        console.log("🎯 [CALIBRATION] No face detected, triggering callbacks");

        this.config.onDistanceUpdate({
          distance: null,
          offset: null,
          distanceRatio: null,
          faceWidth: 0,
          alignmentStatus: "no-face",
          faceDetected: false,
          isCalibrated: this.calibration.isCalibrated,
        });

        // 同时触发校准状态回调
        this.config.onCalibrationFrameStatus({
          status: "no-face",
          faceRatio: 0,
          idealRatio: this.config.calibrationFrame.targetFaceRatio,
          position: {
            x: 0,
            y: 0,
          },
        });
      }
    }

    loadCalibration(calibrationData) {
      try {
        let parsedData = calibrationData;
        if (typeof calibrationData === "string") {
          parsedData = JSON.parse(calibrationData);
        }

        if (
          !parsedData ||
          typeof parsedData !== "object" ||
          parsedData.referenceFaceWidth === undefined ||
          !parsedData.timestamp
        ) {
          console.error("❌ 校准数据格式不正确:", parsedData);
          return false;
        }

        this.calibration = {
          isCalibrated: true,
          referenceFaceWidth: parsedData.referenceFaceWidth,
          referenceDistance: parsedData.referenceDistance || 0,
          referenceFontSize: parsedData.referenceFontSize || 16,
          timestamp: parsedData.timestamp,
        };

        console.log("✅ 已加载校准数据:", this.calibration);
        return true;
      } catch (error) {
        console.error("❌ 加载校准数据失败:", error);
        return false;
      }
    }

    async cleanup() {
      this.isRunning = false;

      if (this.faceMesh) {
        try {
          await this.faceMesh.close();
        } catch (error) {
          console.error("关闭FaceMesh时出错:", error);
        }
      }

      this.stopCamera();
      this.faceHistory = [];
      console.log("✅ MediaPipe Distance Detector 已清理");
    }
  }

  // ===== 阅读模式管理器 =====
  class ReadingModeManager {
    constructor() {
      this.isReadingMode = false;
      this.isDynamicFontEnabled = false;
      this.isDynamicContrastEnabled = false;
      this.distanceDetector = null;
      this.originalContent = null;
      this.readingContainer = null;
      this.floatingButton = null;
      this.floatingMenu = null;

      this.baseFontSize = 16;
      this.currentFontSize = 16;

      this.baseBackgroundColor = "#FBF0D9";
      this.baseTextColor = "#5F4B32";
      this.currentBackgroundColor = "#FBF0D9";
      this.currentTextColor = "#5F4B32";

      this.startTime = null;
      this.timerInterval = null;
      this.timerElement = null;

      this.lastStableDistance = 0;
      this.fontChangeThreshold = 0.5; // 降低阈值使字体变化更灵敏
      this.currentDistance = 0;

      // 死区功能：防止头部小范围变化引起字体变化
      this.deadZoneRadius = 2.0; // 死区半径，在此范围内的变化会被忽略
      this.deadZoneCenter = 0; // 死区中心距离
      this.deadZoneStableTime = 1000; // 需要稳定1秒才更新死区中心 (毫秒)
      this.lastDistanceChangeTime = 0; // 上次距离变化时间

      this.initializeFontSizes();
      this.init();
    }

    initializeFontSizes() {
      try {
        const calibrationData = JSON.parse(
          localStorage.getItem("mediapipe-calibration") || "{}"
        );
        this.baseFontSize = calibrationData.referenceFontSize || 16;
        this.currentFontSize = this.baseFontSize;
        console.log(
          "📖 阅读模式初始化，基准字体大小:",
          this.baseFontSize + "px"
        );
      } catch (error) {
        console.log("📖 未找到校准数据，使用默认字体大小: 16px");
        this.baseFontSize = 16;
        this.currentFontSize = 16;
      }
    }

    init() {
      this.createFloatingButton();
      this.loadSettings();
      this.bindEvents();
    }

    createFloatingButton() {
      const existingButton = document.querySelector(".reading-mode-float-btn");
      const existingMenu = document.querySelector(".reading-mode-float-menu");

      if (existingButton) existingButton.remove();
      if (existingMenu) existingMenu.remove();

      this.floatingButton = document.createElement("div");
      this.floatingButton.className = "reading-mode-float-btn";
      this.floatingButton.innerHTML = `
                <div class="float-btn-icon">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
                        <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
                    </svg>
                </div>
            `;

      this.floatingMenu = document.createElement("div");
      this.floatingMenu.className = "reading-mode-float-menu";
      this.floatingMenu.innerHTML = `
                <div class="float-menu-content">
                    <div class="menu-header">
                        <span class="menu-title">阅读选项</span>
                        <button class="menu-close-btn">×</button>
                    </div>
                    <div class="menu-item">
                        <button class="reading-mode-toggle-btn">
                            <span class="menu-icon">📖</span>
                            <span class="menu-text">进入阅读模式</span>
                        </button>
                    </div>
                    <div class="menu-item">
                        <label class="menu-checkbox">
                            <input type="checkbox" class="dynamic-font-checkbox">
                            <span class="checkmark"></span>
                            <span class="menu-text">动态字体调整</span>
                        </label>
                    </div>
                    <div class="menu-item">
                        <label class="menu-checkbox">
                            <input type="checkbox" class="dynamic-contrast-checkbox">
                            <span class="checkmark"></span>
                            <span class="menu-text">动态对比度调整</span>
                        </label>
                    </div>
                    <div class="menu-item">
                        <button class="calibration-btn">
                            <span class="menu-icon">🎯</span>
                            <span class="menu-text">校准距离</span>
                        </button>
                    </div>
                </div>
            `;

      document.body.appendChild(this.floatingButton);
      document.body.appendChild(this.floatingMenu);
    }

    bindEvents() {
      if (this.globalClickHandler) {
        document.removeEventListener("click", this.globalClickHandler);
      }
      if (this.globalKeyHandler) {
        document.removeEventListener("keydown", this.globalKeyHandler);
      }

      this.floatingButton.addEventListener("click", (e) => {
        e.stopPropagation();
        this.toggleFloatingMenu();
      });

      const closeBtn = this.floatingMenu.querySelector(".menu-close-btn");
      closeBtn.addEventListener("click", () => {
        this.hideFloatingMenu();
      });

      const toggleBtn = this.floatingMenu.querySelector(
        ".reading-mode-toggle-btn"
      );
      toggleBtn.addEventListener("click", () => {
        this.toggleReadingMode();
        this.hideFloatingMenu();
      });

      const dynamicFontCheckbox = this.floatingMenu.querySelector(
        ".dynamic-font-checkbox"
      );
      dynamicFontCheckbox.addEventListener("change", (e) => {
        this.isDynamicFontEnabled = e.target.checked;
        this.saveSettings();

        if (this.isReadingMode) {
          if (this.isDynamicFontEnabled) {
            this.startDynamicFont();
          } else {
            this.stopDynamicFont();
          }
        }
      });

      const dynamicContrastCheckbox = this.floatingMenu.querySelector(
        ".dynamic-contrast-checkbox"
      );
      dynamicContrastCheckbox.addEventListener("change", (e) => {
        console.log(
          `🔧 [CHECKBOX] Dynamic contrast checkbox changed: ${e.target.checked}`
        );
        this.isDynamicContrastEnabled = e.target.checked;
        console.log(
          `🔧 [CHECKBOX] isDynamicContrastEnabled set to: ${this.isDynamicContrastEnabled}`
        );
        this.saveSettings();
        console.log(`🔧 [CHECKBOX] Settings saved`);

        if (this.isReadingMode) {
          console.log(
            `🔧 [CHECKBOX] Reading mode is active, starting/stopping contrast`
          );
          if (this.isDynamicContrastEnabled) {
            console.log(`🔧 [CHECKBOX] Starting dynamic contrast...`);
            this.startDynamicContrast();
          } else {
            console.log(`🔧 [CHECKBOX] Stopping dynamic contrast...`);
            this.stopDynamicContrast();
          }
        } else {
          console.log(
            `🔧 [CHECKBOX] Reading mode is not active, settings saved for later`
          );
        }
      });

      const calibrationBtn =
        this.floatingMenu.querySelector(".calibration-btn");
      calibrationBtn.addEventListener("click", () => {
        this.openCalibrationDialog();
        this.hideFloatingMenu();
      });

      this.globalClickHandler = (e) => {
        if (
          this.floatingMenu &&
          this.floatingButton &&
          !this.floatingMenu.contains(e.target) &&
          !this.floatingButton.contains(e.target)
        ) {
          this.hideFloatingMenu();
        }
      };
      document.addEventListener("click", this.globalClickHandler);

      this.globalKeyHandler = (e) => {
        if (e.key === "Escape" && this.isReadingMode) {
          this.exitReadingMode();
        }
      };
      document.addEventListener("keydown", this.globalKeyHandler);
    }

    toggleFloatingMenu() {
      this.floatingMenu.classList.toggle("show");
      this.floatingButton.classList.toggle("active");
    }

    hideFloatingMenu() {
      this.floatingMenu.classList.remove("show");
      this.floatingButton.classList.remove("active");
    }

    toggleReadingMode() {
      if (this.isReadingMode) {
        this.exitReadingMode();
      } else {
        this.enterReadingMode();
      }
    }

    enterReadingMode() {
      this.isReadingMode = true;
      this.originalContent = document.body.innerHTML;

      const articleContent = this.extractArticleContent();
      this.createReadingModeContainer(articleContent);

      document.body.style.overflow = "hidden";

      const toggleBtn = this.floatingMenu.querySelector(
        ".reading-mode-toggle-btn .menu-text"
      );
      toggleBtn.textContent = "退出阅读模式";

      this.currentBackgroundColor = this.baseBackgroundColor;
      this.currentTextColor = this.baseTextColor;
      this.updateContrast();
      this.updateFontSize();

      this.startReadingTimer();

      if (this.isDynamicFontEnabled) {
        this.startDynamicFont();
      }
      if (this.isDynamicContrastEnabled) {
        this.startDynamicContrast();
      }
    }

    exitReadingMode() {
      this.isReadingMode = false;

      this.stopReadingTimer();
      this.stopDynamicFont();
      this.stopDynamicContrast();

      // 清理动态字体样式
      const dynamicFontStyle = document.getElementById("dynamic-font-style");
      if (dynamicFontStyle) {
        dynamicFontStyle.remove();
        console.log("🧹 [CLEANUP] Removed dynamic font CSS");
      }

      if (this.distanceDetector) {
        console.log("🔴 [EXIT] Force stopping camera detector");
        this.distanceDetector.stopCamera();
        this.distanceDetector = null;
      }

      if (this.readingContainer) {
        this.readingContainer.remove();
        this.readingContainer = null;
      }

      if (this.originalContent) {
        document.body.innerHTML = this.originalContent;
        this.originalContent = null;

        this.floatingButton = null;
        this.floatingMenu = null;

        setTimeout(() => {
          this.init();
        }, 100);
      }

      document.body.style.overflow = "";

      setTimeout(() => {
        const toggleBtn = document.querySelector(
          ".reading-mode-toggle-btn .menu-text"
        );
        if (toggleBtn) {
          toggleBtn.textContent = "进入阅读模式";
        }
      }, 150);
    }

    extractArticleContent() {
      const titleElement = document.querySelector(".title, h1");
      const contentElement = document.querySelector(".content, main, article");

      const title = titleElement ? titleElement.textContent.trim() : "文章标题";
      const content = contentElement
        ? contentElement.innerHTML
        : "<p>未找到文章内容</p>";

      return { title, content };
    }

    createReadingModeContainer(articleContent) {
      const wordCount = this.calculateWordCount(articleContent.content);
      const readingTime = this.calculateReadingTime(wordCount);

      this.readingContainer = document.createElement("div");
      this.readingContainer.className = "reading-mode-container";
      this.readingContainer.innerHTML = `
                <div class="reading-mode-header">
                    <button class="reading-mode-close-btn">×</button>
                    <div class="reading-mode-title">${
                      articleContent.title
                    }</div>
                </div>
                <div class="reading-mode-content">
                    <div class="reading-article-content" id="readingArticleContent">
                        ${articleContent.content}
                    </div>
                </div>
                <div class="reading-mode-footer">
                    <div class="reading-mode-info">
                        <div class="reading-mode-info-left">
                            <span class="font-size-info">字体大小: <span id="currentFontSizeDisplay">${this.currentFontSize.toFixed(
                              1
                            )}px</span></span>
                            ${
                              this.isDynamicFontEnabled
                                ? '<span class="dynamic-font-status">动态字体调整已启用</span>'
                                : ""
                            }
                            ${
                              this.isDynamicContrastEnabled
                                ? '<span class="dynamic-contrast-status">动态对比度调整已启用</span>'
                                : ""
                            }
                        </div>
                        <div class="reading-mode-info-right">
                            <span class="word-count-info">字数: ${wordCount}</span>
                            <span class="reading-time-info">阅读时间: ${readingTime}</span>
                            <span class="reading-timer-info">已阅读: <span id="readingTimer">00:00</span></span>

                        </div>
                    </div>
                </div>
            `;

      document.body.appendChild(this.readingContainer);

      const closeBtn = this.readingContainer.querySelector(
        ".reading-mode-close-btn"
      );
      closeBtn.addEventListener("click", () => {
        this.exitReadingMode();
      });

      // 调试按钮已移除
    }

    calculateWordCount(htmlContent) {
      const tempDiv = document.createElement("div");
      tempDiv.innerHTML = htmlContent;
      const textContent = tempDiv.textContent || tempDiv.innerText || "";
      const chineseChars = (textContent.match(/[\u4e00-\u9fa5]/g) || []).length;
      const englishWords = (textContent.match(/[a-zA-Z]+/g) || []).length;
      return chineseChars + englishWords;
    }

    calculateReadingTime(wordCount) {
      const readingSpeed = 225;
      const minutes = Math.ceil(wordCount / readingSpeed);
      return minutes < 1
        ? "< 1分钟"
        : minutes === 1
        ? "1分钟"
        : `${minutes}分钟`;
    }

    async startDynamicFont() {
      if (!this.isDynamicFontEnabled || !this.isReadingMode) return;

      try {
        const calibrationData = CalibrationManager.getCalibration();
        if (!calibrationData) {
          this.showCalibrationRequiredDialog();
          return;
        }

        console.log(
          "📱 Starting dynamic font with calibration data:",
          calibrationData
        );

        this.distanceDetector = new MediaPipeDistanceDetector({
          onDistanceUpdate: (distance) => this.onDistanceUpdate(distance),
          onError: (error) => this.onDynamicFontError(error),
          onCalibrationFrameStatus: (status) => {
            console.log("📊 Calibration frame status:", status.status);
          },
          smoothingWindow: 5,
          minConfidence: 0.5,
          disableCamera: false,
          basePath: "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/",
        });

        // 先初始化MediaPipe
        await this.distanceDetector.initialize();

        const success = await this.distanceDetector.startCamera();
        if (success) {
          console.log(
            "📥 Loading calibration data into detector:",
            calibrationData
          );
          this.distanceDetector.loadCalibration(calibrationData);

          this.baseFontSize = calibrationData.referenceFontSize || 16;
          this.currentFontSize = this.baseFontSize;

          this.lastStableDistance = 0;
          this.currentDistance = 0;

          this.updateDynamicFontStatus("运行中");
          console.log("✅ Dynamic font started successfully");

          // 启动检测循环
          this.distanceDetector.startDetectionLoop();
        } else {
          throw new Error("无法启动摄像头");
        }
      } catch (error) {
        console.error("❌ Dynamic font startup failed:", error);
        this.onDynamicFontError(error);
      }
    }

    stopDynamicFont() {
      console.log("⏹️ Stopping dynamic font adjustment");

      this.currentFontSize = this.baseFontSize;
      this.updateFontSize();
      this.updateDynamicFontStatus("已停止");

      this.lastStableDistance = 0;
      this.currentDistance = 0;

      // 只有当字体和对比度功能都被禁用时，才停止摄像头检测器
      if (
        !this.isDynamicContrastEnabled &&
        !this.isDynamicFontEnabled &&
        this.distanceDetector
      ) {
        console.log(
          "🔴 [FONT] Stopping camera detector (both features disabled)"
        );
        this.distanceDetector.stopCamera();
        this.distanceDetector = null;
      }
    }

    onDistanceUpdate(distanceData) {
      if (!this.isReadingMode) return;

      console.log("📏 [DISTANCE] Distance update:", distanceData);
      console.log(
        `🔧 [DEBUG] isDynamicFontEnabled: ${this.isDynamicFontEnabled}, isDynamicContrastEnabled: ${this.isDynamicContrastEnabled}`
      );

      this.currentDistance = distanceData.offset;

      if (this.isDynamicFontEnabled) {
        console.log("📝 [FONT] Processing font adjustment...");
        console.log(
          `📝 [FONT] Current font size before adjustment: ${this.currentFontSize}px`
        );
        this.handleStableFontAdjustment(distanceData.offset);
        console.log(
          `📝 [FONT] Current font size after adjustment: ${this.currentFontSize}px`
        );
      }

      if (this.isDynamicContrastEnabled) {
        console.log("🎨 [CONTRAST] Processing contrast adjustment...");
        const targetRatio = this.calculateContrastAdjustment(
          distanceData.offset
        );

        console.log(
          `🎨 [CONTRAST] Contrast calculation: offset=${distanceData.offset.toFixed(
            2
          )}, targetRatio=${targetRatio.toFixed(2)}:1`
        );

        const contrastColors = this.getContrastColorsForRatio(targetRatio);

        const prevBgColor = this.currentBackgroundColor;
        const prevTextColor = this.currentTextColor;

        this.currentBackgroundColor = contrastColors.background;
        this.currentTextColor = contrastColors.text;

        console.log(
          `🎨 [CONTRAST] Color change: bg ${prevBgColor} → ${this.currentBackgroundColor}, text ${prevTextColor} → ${this.currentTextColor}`
        );
        console.log(
          `🎨 [CONTRAST] Smooth contrast: ${contrastColors.ratio.toFixed(
            2
          )}:1 (target: ${targetRatio.toFixed(2)}:1)`
        );

        this.updateContrast();
      }

      if (Date.now() % 2000 < 50) {
        const logMessages = [];
        if (this.isDynamicFontEnabled) {
          logMessages.push(`Font: ${this.currentFontSize.toFixed(1)}px`);
        }
        if (this.isDynamicContrastEnabled) {
          logMessages.push(
            `Contrast: ${this.currentBackgroundColor} / ${this.currentTextColor}`
          );
        }
        if (logMessages.length > 0) {
          console.log(
            `📊 [SUMMARY] ${logMessages.join(
              ", "
            )} (offset: ${distanceData.offset.toFixed(1)})`
          );
        }
      }
    }

    handleStableFontAdjustment(distance) {
      console.log(
        `🔍 [FONT-ADJ] Starting font adjustment: distance=${distance}, lastStableDistance=${this.lastStableDistance}`
      );

      // 死区逻辑：检查是否在死区范围内
      const distanceFromDeadZoneCenter = Math.abs(
        distance - this.deadZoneCenter
      );
      const currentTime = Date.now();

      console.log(
        `⚡ [DEADZONE] Distance=${distance.toFixed(
          2
        )}, DeadZoneCenter=${this.deadZoneCenter.toFixed(
          2
        )}, DistanceFromCenter=${distanceFromDeadZoneCenter.toFixed(
          2
        )}, DeadZoneRadius=${this.deadZoneRadius}`
      );

      // 如果在死区内，不进行字体调整
      if (distanceFromDeadZoneCenter <= this.deadZoneRadius) {
        console.log(
          `🚫 [DEADZONE] Distance within dead zone (${distanceFromDeadZoneCenter.toFixed(
            2
          )} <= ${this.deadZoneRadius}), skipping adjustment`
        );
        return;
      }

      // 更新死区中心（如果距离变化稳定一段时间）
      if (currentTime - this.lastDistanceChangeTime > this.deadZoneStableTime) {
        const oldCenter = this.deadZoneCenter;
        this.deadZoneCenter = distance;
        console.log(
          `🎯 [DEADZONE] Updated dead zone center: ${oldCenter.toFixed(
            2
          )} → ${this.deadZoneCenter.toFixed(2)}`
        );
      }
      this.lastDistanceChangeTime = currentTime;

      const distanceChange = Math.abs(distance - this.lastStableDistance);
      console.log(
        `🔍 [FONT-ADJ] Distance change: ${distanceChange}, threshold: ${this.fontChangeThreshold}`
      );

      if (distanceChange >= this.fontChangeThreshold) {
        console.log(
          `✅ [FONT-ADJ] Distance change sufficient, calculating new font size...`
        );

        const fontSizeMultiplier = this.calculateFontSizeMultiplier(distance);
        console.log(
          `🔍 [FONT-ADJ] Font multiplier: ${fontSizeMultiplier}, baseFontSize: ${this.baseFontSize}`
        );

        const newFontSize = Math.max(
          12,
          Math.min(32, this.baseFontSize * fontSizeMultiplier)
        );
        console.log(
          `🔍 [FONT-ADJ] New font size calculated: ${newFontSize}, current: ${this.currentFontSize}`
        );

        const fontSizeChange = Math.abs(newFontSize - this.currentFontSize);
        console.log(
          `🔍 [FONT-ADJ] Font size change: ${fontSizeChange}, minimum required: 0.2`
        );

        if (fontSizeChange >= 0.2) {
          console.log(
            `✅ [FONT-ADJ] Font size change sufficient, updating font...`
          );
          this.currentFontSize = newFontSize;
          this.updateFontSize();

          this.lastStableDistance = distance;

          const fontSizeDisplay = document.getElementById(
            "currentFontSizeDisplay"
          );
          if (fontSizeDisplay) {
            fontSizeDisplay.textContent = `${this.currentFontSize.toFixed(
              1
            )}px`;
          }

          console.log(
            `📝 [FONT] Font size updated to: ${this.currentFontSize.toFixed(
              1
            )}px (distance change: ${distanceChange.toFixed(2)}, threshold: ${
              this.fontChangeThreshold
            })`
          );
        } else {
          console.log(
            `❌ [FONT-ADJ] Font size change too small (${fontSizeChange}), skipping update`
          );
        }
      } else {
        console.log(
          `❌ [FONT-ADJ] Distance change too small (${distanceChange}), skipping adjustment`
        );
      }
    }

    injectDynamicFontCSS(fontSize) {
      // 移除之前的动态字体样式
      const existingStyle = document.getElementById("dynamic-font-style");
      if (existingStyle) {
        existingStyle.remove();
      }

      // 创建新的动态字体样式，使用最高优先级
      const style = document.createElement("style");
      style.id = "dynamic-font-style";
      style.textContent = `
        /* 动态字体样式 - 最高优先级 */
        #readingArticleContent,
        #readingArticleContent *,
        .reading-article-content,
        .reading-article-content * {
          font-size: ${fontSize}px !important;
        }
        
        /* 特别针对常见文本元素 */
        #readingArticleContent p,
        #readingArticleContent span,
        #readingArticleContent div,
        #readingArticleContent a,
        #readingArticleContent strong,
        #readingArticleContent em,
        #readingArticleContent b,
        #readingArticleContent i,
        .reading-article-content p,
        .reading-article-content span,
        .reading-article-content div,
        .reading-article-content a,
        .reading-article-content strong,
        .reading-article-content em,
        .reading-article-content b,
        .reading-article-content i {
          font-size: ${fontSize}px !important;
        }
        
        /* 标题元素使用相对比例 */
        #readingArticleContent h1,
        .reading-article-content h1 {
          font-size: ${fontSize * 1.5}px !important;
        }
        
        #readingArticleContent h2,
        .reading-article-content h2 {
          font-size: ${fontSize * 1.3}px !important;
        }
        
        #readingArticleContent h3,
        .reading-article-content h3 {
          font-size: ${fontSize * 1.1}px !important;
        }
        
        #readingArticleContent h4,
        #readingArticleContent h5,
        #readingArticleContent h6,
        .reading-article-content h4,
        .reading-article-content h5,
        .reading-article-content h6 {
          font-size: ${fontSize}px !important;
        }
      `;

      // 插入到head中，确保高优先级
      document.head.appendChild(style);

      console.log(
        `🎨 [FONT] Injected dynamic CSS with font-size: ${fontSize}px`
      );
    }

    calculateFontSizeMultiplier(distance) {
      const scale = 1.5;
      const multiplier = 1 + (distance * scale * 0.8) / 100;
      const result = Math.max(0.6, Math.min(2.5, multiplier));

      console.log(
        `📝 [FONT] Distance=${distance.toFixed(2)}, multiplier=${result.toFixed(
          2
        )} (${
          distance > 0 ? "更远→更大" : distance < 0 ? "更近→更小" : "基准"
        })`
      );

      return result;
    }

    updateFontSize() {
      console.log(
        `🔍 [FONT] updateFontSize called with size: ${this.currentFontSize}px`
      );

      const contentElement = document.getElementById("readingArticleContent");
      console.log(`🔍 [FONT] contentElement found:`, contentElement);

      if (contentElement) {
        // 记录设置前的字体大小
        const beforeSize = window.getComputedStyle(contentElement).fontSize;
        console.log(
          `🔍 [FONT] Before setting - computed fontSize: ${beforeSize}`
        );

        // 设置容器的字体大小，使用多重策略强制覆盖网站CSS
        contentElement.style.transition = "font-size 0.3s ease";

        // 方法1: 直接设置style属性，最高优先级
        contentElement.style.setProperty(
          "font-size",
          `${this.currentFontSize}px`,
          "important"
        );

        // 方法2: 设置所有文本元素的字体大小
        const textElements = contentElement.querySelectorAll("*");
        textElements.forEach((element) => {
          // 只对文本元素设置字体大小，避免影响按钮等UI元素
          const tagName = element.tagName.toLowerCase();
          if (
            [
              "p",
              "span",
              "div",
              "a",
              "strong",
              "em",
              "b",
              "i",
              "small",
              "label",
              "li",
              "h1",
              "h2",
              "h3",
              "h4",
              "h5",
              "h6",
            ].includes(tagName)
          ) {
            element.style.setProperty(
              "font-size",
              `${this.currentFontSize}px`,
              "important"
            );
          }
        });

        // 方法3: 通过动态CSS规则强制覆盖
        this.injectDynamicFontCSS(this.currentFontSize);

        // 验证设置是否成功
        setTimeout(() => {
          const afterSize = window.getComputedStyle(contentElement).fontSize;
          console.log(
            `🔍 [FONT] After setting - computed fontSize: ${afterSize}`
          );
          console.log(
            `🔍 [FONT] Element style fontSize: ${contentElement.style.fontSize}`
          );
        }, 50);

        console.log(
          `📝 [FONT] Updated font size to ${this.currentFontSize.toFixed(
            1
          )}px with !important`
        );
      } else {
        console.error("❌ [FONT] readingArticleContent element not found!");
        // 尝试查找其他可能的元素
        const allElements = document.querySelectorAll(
          '[id*="reading"], [class*="reading"], [id*="article"], [class*="article"]'
        );
        console.log("🔍 [FONT] Found reading/article elements:", allElements);
      }

      // 更新字体大小显示
      const fontSizeDisplay = document.getElementById("currentFontSizeDisplay");
      if (fontSizeDisplay) {
        fontSizeDisplay.textContent = `${this.currentFontSize.toFixed(1)}px`;
      }
    }

    onDynamicFontError(error) {
      console.error("Dynamic font error:", error);
      this.onDynamicError(error);
    }

    updateDynamicFontStatus(status) {
      const statusElement = document.querySelector(".dynamic-font-status");
      if (statusElement) {
        statusElement.textContent = `动态字体: ${status}`;
      }
    }

    async startDynamicContrast() {
      if (!this.isDynamicContrastEnabled || !this.isReadingMode) return;

      console.log("🎨 [CONTRAST] Starting dynamic contrast adjustment");

      this.currentBackgroundColor = this.baseBackgroundColor;
      this.currentTextColor = this.baseTextColor;

      this.updateContrast();

      if (!this.distanceDetector) {
        try {
          const calibrationData = CalibrationManager.getCalibration();

          if (!calibrationData) {
            this.showCalibrationRequiredDialog();
            return;
          }

          this.distanceDetector = new MediaPipeDistanceDetector({
            onDistanceUpdate: (distance) => this.onDistanceUpdate(distance),
            onError: (error) => this.onDynamicError(error),
            onCalibrationFrameStatus: (status) => {
              console.log(
                "📊 [CONTRAST] Calibration frame status:",
                status.status
              );
            },
            smoothingWindow: 5,
            minConfidence: 0.5,
            disableCamera: false,
            basePath: "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/",
          });

          await this.distanceDetector.initialize();

          const success = await this.distanceDetector.startCamera();

          if (success) {
            this.distanceDetector.loadCalibration(calibrationData);
            this.distanceDetector.startDetectionLoop();
            console.log("✅ [CONTRAST] Dynamic contrast started successfully");
          } else {
            throw new Error("无法启动摄像头");
          }
        } catch (error) {
          console.error(
            "❌ [CONTRAST] Dynamic contrast startup failed:",
            error
          );
          this.onDynamicError(error);
        }
      }

      this.updateDynamicContrastStatus("运行中");
    }

    stopDynamicContrast() {
      console.log("⏹️ Stopping dynamic contrast adjustment");

      this.currentBackgroundColor = this.baseBackgroundColor;
      this.currentTextColor = this.baseTextColor;
      this.updateContrast();
      this.updateDynamicContrastStatus("已停止");

      // 只有当字体和对比度功能都被禁用时，才停止摄像头检测器
      if (
        !this.isDynamicFontEnabled &&
        !this.isDynamicContrastEnabled &&
        this.distanceDetector
      ) {
        console.log(
          "🔴 [CONTRAST] Stopping camera detector (both features disabled)"
        );
        this.distanceDetector.stopCamera();
        this.distanceDetector = null;
      }
    }

    calculateContrastAdjustment(distance) {
      const scale = 2.0;
      const normalizedDistance = (distance * scale) / 100;

      const minRatio = 3.5;
      const maxRatio = 9.0;
      const baseRatio = 6.0;

      let targetRatio;
      if (normalizedDistance <= 0) {
        const factor = Math.max(-1, normalizedDistance / 0.5);
        targetRatio = baseRatio + (minRatio - baseRatio) * Math.abs(factor);
      } else {
        const factor = Math.min(1, normalizedDistance / 0.5);
        targetRatio = baseRatio + (maxRatio - baseRatio) * factor;
      }

      targetRatio = Math.max(minRatio, Math.min(maxRatio, targetRatio));

      console.log(
        `🔢 [CONTRAST] Smooth calculation: distance=${distance.toFixed(
          2
        )}, normalized=${normalizedDistance.toFixed(
          2
        )}, targetRatio=${targetRatio.toFixed(2)}:1`
      );

      return targetRatio;
    }

    updateContrast() {
      if (!this.isReadingMode) return;

      const container = document.querySelector(".reading-mode-container");
      if (!container) {
        console.log("🔍 [CONTRAST] Container not found");
        return;
      }

      console.log(
        "🎨 [CONTRAST] Updating contrast:",
        this.currentBackgroundColor,
        this.currentTextColor
      );

      // First update the main container itself
      container.style.setProperty(
        "background-color",
        this.currentBackgroundColor,
        "important"
      );
      container.style.setProperty(
        "background",
        this.currentBackgroundColor,
        "important"
      );

      const backgroundElements = [
        ".reading-mode-header",
        ".reading-mode-content",
        ".reading-mode-footer",
      ];

      console.log(`🎨 [CONTRAST] Updating container and child elements`);

      backgroundElements.forEach((selector) => {
        const elements = container.querySelectorAll(selector);
        console.log(
          `🎨 [CONTRAST] Updating ${elements.length} elements for ${selector}`
        );
        elements.forEach((element) => {
          element.style.setProperty(
            "background-color",
            this.currentBackgroundColor,
            "important"
          );
          element.style.setProperty(
            "background",
            this.currentBackgroundColor,
            "important"
          );
        });
      });

      const textElements = [
        ".reading-mode-title",
        ".reading-mode-close-btn",
        ".reading-mode-header",
        ".reading-mode-header *",
        ".reading-article-content",
        ".reading-article-content *",
        ".reading-mode-footer",
        ".reading-mode-footer *",
        ".reading-mode-info",
        ".reading-mode-info-left",
        ".reading-mode-info-right",
        ".font-size-info",
        ".word-count-info",
        ".reading-time-info",
        ".dynamic-font-status",
        ".dynamic-contrast-status",
      ];

      textElements.forEach((selector) => {
        const elements = container.querySelectorAll(selector);
        console.log(
          `🔤 [CONTRAST] Updating text for ${elements.length} elements: ${selector}`
        );
        elements.forEach((element) => {
          element.style.setProperty(
            "color",
            this.currentTextColor,
            "important"
          );
        });
      });

      const allElements = container.querySelectorAll("*");
      console.log(
        `🔤 [CONTRAST] Updating ${allElements.length} total child elements`
      );
      allElements.forEach((element) => {
        if (
          !element.classList.contains("menu-close-btn") &&
          !element.classList.contains("float-btn-icon")
        ) {
          element.style.setProperty(
            "color",
            this.currentTextColor,
            "important"
          );
        }
      });

      console.log("✅ [CONTRAST] Contrast update completed");

      setTimeout(() => {
        const containerStyle = window.getComputedStyle(container);
        const titleElement = container.querySelector(".reading-mode-title");
        const contentElement = container.querySelector(
          ".reading-article-content"
        );

        console.log(
          `🔍 [VERIFY] Container computed bg: ${containerStyle.backgroundColor}`
        );
        console.log(
          `🔍 [VERIFY] Container computed color: ${containerStyle.color}`
        );

        if (titleElement) {
          const titleStyle = window.getComputedStyle(titleElement);
          console.log(`🔍 [VERIFY] Title computed color: ${titleStyle.color}`);
        }

        if (contentElement) {
          const contentStyle = window.getComputedStyle(contentElement);
          console.log(
            `🔍 [VERIFY] Content computed color: ${contentStyle.color}`
          );
        }
      }, 100);
    }

    calculateRelativeLuminance(hex) {
      const r = parseInt(hex.substr(1, 2), 16) / 255;
      const g = parseInt(hex.substr(3, 2), 16) / 255;
      const b = parseInt(hex.substr(5, 2), 16) / 255;

      const rLinear =
        r <= 0.03928 ? r / 12.92 : Math.pow((r + 0.055) / 1.055, 2.4);
      const gLinear =
        g <= 0.03928 ? g / 12.92 : Math.pow((g + 0.055) / 1.055, 2.4);
      const bLinear =
        b <= 0.03928 ? b / 12.92 : Math.pow((b + 0.055) / 1.055, 2.4);

      return 0.2126 * rLinear + 0.7152 * gLinear + 0.0722 * bLinear;
    }

    calculateContrastRatio(color1, color2) {
      const l1 = this.calculateRelativeLuminance(color1);
      const l2 = this.calculateRelativeLuminance(color2);

      const lighter = Math.max(l1, l2);
      const darker = Math.min(l1, l2);

      return (lighter + 0.05) / (darker + 0.05);
    }

    adjustColorForContrast(
      baseColor,
      targetColor,
      targetRatio,
      adjustBackground = true
    ) {
      const baseLuminance = this.calculateRelativeLuminance(baseColor);
      const targetLuminance = this.calculateRelativeLuminance(targetColor);

      let newLuminance;
      if (adjustBackground) {
        if (baseLuminance > targetLuminance) {
          newLuminance = (baseLuminance + 0.05) * targetRatio - 0.05;
        } else {
          newLuminance = (baseLuminance + 0.05) / targetRatio - 0.05;
        }
      } else {
        if (targetLuminance > baseLuminance) {
          newLuminance = (targetLuminance + 0.05) / targetRatio - 0.05;
        } else {
          newLuminance = (targetLuminance + 0.05) * targetRatio - 0.05;
        }
      }

      newLuminance = Math.max(0, Math.min(1, newLuminance));

      return this.luminanceToColor(newLuminance, targetColor);
    }

    luminanceToColor(targetLuminance, baseColor) {
      targetLuminance = Math.max(0, Math.min(1, targetLuminance));

      const r = parseInt(baseColor.substr(1, 2), 16);
      const g = parseInt(baseColor.substr(3, 2), 16);
      const b = parseInt(baseColor.substr(5, 2), 16);

      const currentLuminance = this.calculateRelativeLuminance(baseColor);

      if (Math.abs(targetLuminance - currentLuminance) < 0.001) {
        return baseColor;
      }

      let factor;
      if (targetLuminance > currentLuminance) {
        factor = Math.pow(targetLuminance / currentLuminance, 0.4);
      } else {
        factor = Math.pow(targetLuminance / currentLuminance, 0.6);
      }

      const newR = Math.round(Math.max(0, Math.min(255, r * factor)));
      const newG = Math.round(Math.max(0, Math.min(255, g * factor)));
      const newB = Math.round(Math.max(0, Math.min(255, b * factor)));

      const result = `#${newR.toString(16).padStart(2, "0")}${newG
        .toString(16)
        .padStart(2, "0")}${newB.toString(16).padStart(2, "0")}`;

      console.log(
        `🎨 [COLOR] Luminance conversion: ${baseColor} (${currentLuminance.toFixed(
          3
        )}) → ${result} (${this.calculateRelativeLuminance(result).toFixed(
          3
        )}) target: ${targetLuminance.toFixed(3)}`
      );

      return result;
    }

    getContrastColorsForRatio(targetRatio) {
      const baseBackground = this.baseBackgroundColor;
      const baseText = this.baseTextColor;

      const adjustedText = this.adjustColorForContrast(
        baseText,
        baseBackground,
        targetRatio,
        false
      );

      let adjustedBackground = baseBackground;
      const baseRatio = 6.0;

      if (targetRatio < baseRatio) {
        // 低对比度：调亮背景
        const factor = 1 + (baseRatio - targetRatio) * 0.03; // 增加变化幅度
        adjustedBackground = this.adjustBrightness(baseBackground, factor);
      } else if (targetRatio > baseRatio) {
        // 高对比度：调暗背景
        const factor = 1 - (targetRatio - baseRatio) * 0.02; // 增加变化幅度
        adjustedBackground = this.adjustBrightness(baseBackground, factor);
      }

      const finalRatio = this.calculateContrastRatio(
        adjustedText,
        adjustedBackground
      );

      let description;
      if (finalRatio < 4.5) {
        description = "低对比度 (未达标)";
      } else if (finalRatio < 7.0) {
        description = "中等对比度 (AA标准)";
      } else {
        description = "高对比度 (AAA标准)";
      }

      console.log(
        `🎨 [CONTRAST] Smooth ratio: target=${targetRatio.toFixed(
          2
        )}:1, actual=${finalRatio.toFixed(2)}:1 (${description})`
      );

      return {
        background: adjustedBackground,
        text: adjustedText,
        ratio: finalRatio,
        description: description,
      };
    }

    adjustBrightness(hex, factor) {
      const r = parseInt(hex.substr(1, 2), 16);
      const g = parseInt(hex.substr(3, 2), 16);
      const b = parseInt(hex.substr(5, 2), 16);

      const newR = Math.round(Math.max(0, Math.min(255, r * factor)));
      const newG = Math.round(Math.max(0, Math.min(255, g * factor)));
      const newB = Math.round(Math.max(0, Math.min(255, b * factor)));

      return `#${newR.toString(16).padStart(2, "0")}${newG
        .toString(16)
        .padStart(2, "0")}${newB.toString(16).padStart(2, "0")}`;
    }

    updateDynamicContrastStatus(status) {
      const statusElement = document.querySelector(".dynamic-contrast-status");
      if (statusElement) {
        statusElement.textContent = `动态对比度: ${status}`;
      }
    }

    openCalibrationDialog() {
      // Create complete calibration dialog
      const dialog = document.createElement("div");
      dialog.className = "calibration-dialog-overlay";
      dialog.innerHTML = `
        <div class="calibration-dialog-full">
          <div class="calibration-dialog-header">
            <h3>📏 字体校准</h3>
            <button class="calibration-close-btn">×</button>
          </div>
          
          <div class="calibration-dialog-body">
            <div class="calibration-description">
              <p>请将您的脸部对准摄像头，保持在方形框内，系统将自动校准您的阅读距离。</p>
            </div>
            
            <div class="calibration-container">
              <div class="calibration-left">
                <div class="video-container">
                  <video id="calibrationVideo" class="video-preview" autoplay muted></video>
                  <div class="calibration-overlay" id="calibrationOverlay">
                    <div class="calibration-instruction" id="calibrationInstruction">
                      请将脸部对准框内
                    </div>
                  </div>
                </div>
              </div>
              
              <div class="calibration-right">
                <div class="status-indicator waiting" id="calibrationStatus">
                  <div class="status-dot"></div>
                  <span id="statusText">等待摄像头启动...</span>
                </div>
                
                <div class="control-group">
                  <label class="control-label">基准字体大小</label>
                  <div class="slider-container">
                    <input type="range" id="baseFontSize" class="slider" min="10" max="32" value="16" step="0.5" />
                    <span class="slider-value" id="baseFontValue">16px</span>
                  </div>
                  <div class="control-hint">
                    设置阅读模式的基准字体大小。动态调整范围：12px - 32px
                  </div>
                </div>
                
                <div class="control-group">
                  <label class="control-label">字体预览</label>
                  <div class="font-preview" id="fontPreview">
                    这是基准字体大小的预览效果
                  </div>
                </div>
                
                <div class="calibration-buttons">
                  <button id="startCameraBtn" class="btn btn-secondary">启动摄像头</button>
                  <button id="startCalibrationBtn" class="btn btn-primary" disabled>开始校准</button>
                  <button id="saveCalibrationBtn" class="btn btn-primary" disabled>保存校准</button>
                  <button id="resetCalibrationBtn" class="btn btn-danger">重置校准</button>
                </div>
              </div>
            </div>
            
            <div class="calibration-info">
              <h4>📖 校准说明</h4>
              <ul>
                <li>确保您的脸部完全在方形框内</li>
                <li>保持正常的阅读距离（约50-70厘米）</li>
                <li>调整基准字体大小到您感觉舒适的大小</li>
                <li>点击"开始校准"按钮，保持姿势3秒钟</li>
                <li>校准完成后点击"保存校准"</li>
              </ul>
            </div>
          </div>
        </div>
      `;

      document.body.appendChild(dialog);

      // Initialize calibration manager
      const calibrationManager = new CalibrationDialogManager(dialog);

      // Handle dialog close
      const closeBtn = dialog.querySelector(".calibration-close-btn");
      closeBtn.addEventListener("click", () => {
        calibrationManager.cleanup();
        dialog.remove();
      });

      dialog.addEventListener("click", (e) => {
        if (e.target === dialog) {
          calibrationManager.cleanup();
          dialog.remove();
        }
      });
    }

    showCalibrationRequiredDialog() {
      console.log("⚠️ Showing calibration required dialog");

      const calibrationData = CalibrationManager.getCalibration();
      let message = "请先在校准页面进行校准后再使用动态字体调整功能";

      if (calibrationData) {
        if (!calibrationData.referenceFaceWidth) {
          message = "校准数据不完整，请重新进行校准";
        } else {
          message = "校准数据可能已过期，请重新进行校准";
        }
      }

      this.showErrorMessage(message);

      const checkbox = this.floatingMenu.querySelector(
        ".dynamic-font-checkbox"
      );
      if (checkbox) {
        checkbox.checked = false;
        this.isDynamicFontEnabled = false;
        this.saveSettings();
      }
    }

    showErrorMessage(message) {
      const toast = document.createElement("div");
      toast.className = "error-toast";
      toast.textContent = message;

      document.body.appendChild(toast);

      setTimeout(() => {
        toast.remove();
      }, 3000);
    }

    loadSettings() {
      console.log(`🔧 [LOAD-SETTINGS] Loading settings from localStorage...`);
      const settings = localStorage.getItem("reading-mode-settings");
      console.log(`🔧 [LOAD-SETTINGS] Raw settings:`, settings);

      if (settings) {
        const parsed = JSON.parse(settings);
        console.log(`🔧 [LOAD-SETTINGS] Parsed settings:`, parsed);

        this.isDynamicFontEnabled = parsed.isDynamicFontEnabled || false;
        this.isDynamicContrastEnabled =
          parsed.isDynamicContrastEnabled || false;

        console.log(
          `🔧 [LOAD-SETTINGS] Set isDynamicFontEnabled: ${this.isDynamicFontEnabled}`
        );
        console.log(
          `🔧 [LOAD-SETTINGS] Set isDynamicContrastEnabled: ${this.isDynamicContrastEnabled}`
        );

        const dynamicFontCheckbox = this.floatingMenu.querySelector(
          ".dynamic-font-checkbox"
        );
        if (dynamicFontCheckbox) {
          dynamicFontCheckbox.checked = this.isDynamicFontEnabled;
          console.log(
            `🔧 [LOAD-SETTINGS] Font checkbox set to: ${dynamicFontCheckbox.checked}`
          );
        }
        const dynamicContrastCheckbox = this.floatingMenu.querySelector(
          ".dynamic-contrast-checkbox"
        );
        if (dynamicContrastCheckbox) {
          dynamicContrastCheckbox.checked = this.isDynamicContrastEnabled;
          console.log(
            `🔧 [LOAD-SETTINGS] Contrast checkbox set to: ${dynamicContrastCheckbox.checked}`
          );
        }
      } else {
        console.log(`🔧 [LOAD-SETTINGS] No settings found, using defaults`);
      }
    }

    saveSettings() {
      const settings = {
        isDynamicFontEnabled: this.isDynamicFontEnabled,
        isDynamicContrastEnabled: this.isDynamicContrastEnabled,
      };
      localStorage.setItem("reading-mode-settings", JSON.stringify(settings));
    }

    onDynamicError(error) {
      console.error("❌ Dynamic features error:", error);

      this.updateDynamicFontStatus("错误");
      this.updateDynamicContrastStatus("错误");

      const errorMessage = error.message || "动态功能启动失败";
      const statusContainer = document.querySelector(".reading-mode-info-left");
      if (statusContainer) {
        const errorSpan = document.createElement("span");
        errorSpan.className = "dynamic-error-status";
        errorSpan.style.color = "#ff4444";
        errorSpan.textContent = `错误: ${errorMessage}`;
        statusContainer.appendChild(errorSpan);

        setTimeout(() => {
          if (errorSpan.parentNode) {
            errorSpan.parentNode.removeChild(errorSpan);
          }
        }, 5000);
      }
    }

    startReadingTimer() {
      this.startTime = Date.now();
      this.timerElement = document.getElementById("readingTimer");

      if (this.timerElement) {
        this.timerInterval = setInterval(() => {
          this.updateTimerDisplay();
        }, 1000);

        console.log("⏱️ Reading timer started");
      }
    }

    stopReadingTimer() {
      if (this.timerInterval) {
        clearInterval(this.timerInterval);
        this.timerInterval = null;
      }

      if (this.startTime) {
        const readingDuration = Date.now() - this.startTime;

        // 只显示阅读时长，不保存下载文件
        console.log(
          `⏱️ Reading timer stopped. Duration: ${this.formatTime(
            readingDuration
          )}`
        );

        this.startTime = null;
        this.timerElement = null;
      }
    }

    updateTimerDisplay() {
      if (this.startTime && this.timerElement) {
        const elapsed = Date.now() - this.startTime;
        this.timerElement.textContent = this.formatTime(elapsed);
      }
    }

    formatTime(milliseconds) {
      const totalSeconds = Math.floor(milliseconds / 1000);
      const minutes = Math.floor(totalSeconds / 60);
      const seconds = totalSeconds % 60;
      return `${minutes.toString().padStart(2, "0")}:${seconds
        .toString()
        .padStart(2, "0")}`;
    }

    getCurrentArticleNumber() {
      const urlMatch = window.location.pathname.match(/article-(\d+)\.html/);
      if (urlMatch) {
        return urlMatch[1];
      }

      const activeMenuItem = document.querySelector(".menu-item.active");
      if (activeMenuItem) {
        const hrefMatch = activeMenuItem.href.match(/article-(\d+)\.html/);
        if (hrefMatch) {
          return hrefMatch[1];
        }
      }

      const titleMatch = document.title.match(/^(\d+)\./);
      if (titleMatch) {
        return titleMatch[1];
      }

      return "unknown";
    }

    saveReadingTime(duration) {
      const articleNumber = this.getCurrentArticleNumber();
      const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
      const filename = `article-${articleNumber}-${timestamp}.txt`;
      const readingTimeFormatted = this.formatTime(duration);

      const content = `Article: ${articleNumber}
Reading Duration: ${readingTimeFormatted}
Start Time: ${new Date(this.startTime).toLocaleString()}
End Time: ${new Date().toLocaleString()}
Total Seconds: ${Math.floor(duration / 1000)}`;

      try {
        const blob = new Blob([content], { type: "text/plain" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        a.style.display = "none";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        console.log(`💾 Reading time saved to ${filename}`);

        this.showSuccessMessage(`阅读时间已保存: ${readingTimeFormatted}`);
      } catch (error) {
        console.error("❌ Error saving reading time:", error);
        this.showErrorMessage("保存阅读时间失败");
      }
    }

    showSuccessMessage(message) {
      const toast = document.createElement("div");
      toast.className = "success-toast";
      toast.textContent = message;

      document.body.appendChild(toast);

      setTimeout(() => {
        toast.remove();
      }, 3000);
    }
  }

  // ===== 校准对话框管理器 =====
  class CalibrationDialogManager {
    constructor(dialogElement) {
      this.dialog = dialogElement;
      this.videoElement = dialogElement.querySelector("#calibrationVideo");
      this.statusIndicator = dialogElement.querySelector("#calibrationStatus");
      this.statusText = dialogElement.querySelector("#statusText");
      this.fontPreview = dialogElement.querySelector("#fontPreview");
      this.baseFontSize = dialogElement.querySelector("#baseFontSize");
      this.baseFontValue = dialogElement.querySelector("#baseFontValue");
      this.startCameraBtn = dialogElement.querySelector("#startCameraBtn");
      this.startCalibrationBtn = dialogElement.querySelector(
        "#startCalibrationBtn"
      );
      this.saveCalibrationBtn = dialogElement.querySelector(
        "#saveCalibrationBtn"
      );
      this.resetCalibrationBtn = dialogElement.querySelector(
        "#resetCalibrationBtn"
      );
      this.calibrationOverlay = dialogElement.querySelector(
        "#calibrationOverlay"
      );
      this.calibrationInstruction = dialogElement.querySelector(
        "#calibrationInstruction"
      );

      this.distanceDetector = null;
      this.isCalibrating = false;
      this.calibrationData = null;
      this.currentDistance = 0;
      this.baseFontSizeValue = 16;

      this.init();
    }

    init() {
      // Base font size slider
      this.baseFontSize.addEventListener("input", (e) => {
        this.baseFontSizeValue = parseFloat(e.target.value);
        this.baseFontValue.textContent = this.baseFontSizeValue + "px";
        this.updateFontPreview();
      });

      // Button events
      this.startCameraBtn.addEventListener("click", () => this.startCamera());
      this.startCalibrationBtn.addEventListener("click", () =>
        this.startCalibration()
      );
      this.saveCalibrationBtn.addEventListener("click", () =>
        this.saveCalibration()
      );
      this.resetCalibrationBtn.addEventListener("click", () =>
        this.resetCalibration()
      );

      // Check existing calibration
      this.checkExistingCalibration();

      // Initialize button states
      this.startCameraBtn.disabled = false;
      this.startCalibrationBtn.disabled = true;
      this.saveCalibrationBtn.disabled = true;

      console.log("🎯 Calibration dialog initialized");
    }

    async startCamera() {
      try {
        this.startCameraBtn.disabled = true;
        this.startCameraBtn.textContent = "启动中...";
        this.updateStatus("detecting", "正在启动摄像头...");

        this.distanceDetector = new MediaPipeDistanceDetector({
          onDistanceUpdate: (distance) => this.onDistanceUpdate(distance),
          onError: (error) => this.onError(error),
          onCalibrationFrameStatus: (status) =>
            this.onCalibrationFrameStatus(status),
          smoothingWindow: 5,
          minConfidence: 0.5,
          disableCamera: false,
          basePath: "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/",
        });

        // 先初始化MediaPipe
        await this.distanceDetector.initialize();

        const success = await this.distanceDetector.startCamera();

        if (success) {
          this.distanceDetector.statusThrottleDelay = 50;
          this.distanceDetector.statusStabilityThreshold = 1;

          this.videoElement.srcObject = this.distanceDetector.video.srcObject;
          this.updateStatus("aligned", "摄像头已启动，请调整位置");

          this.calibrationOverlay.classList.add("active", "no-face");
          this.calibrationInstruction.textContent = "请将脸部对准框内";

          this.startCameraBtn.textContent = "摄像头已启动";
          this.startCameraBtn.disabled = true;

          // 启动MediaPipe检测循环
          this.distanceDetector.startDetectionLoop();

          console.log("✅ Camera started successfully in calibration dialog");
        } else {
          throw new Error("摄像头启动失败");
        }
      } catch (error) {
        console.error("❌ Camera initialization failed:", error);
        this.updateStatus("error", "摄像头启动失败，请检查权限设置");
        this.showToast("摄像头启动失败，请检查权限设置", "error");

        this.startCameraBtn.disabled = false;
        this.startCameraBtn.textContent = "启动摄像头";
      }
    }

    onDistanceUpdate(distanceData) {
      if (!distanceData) return;
      this.currentDistance = distanceData.offset;
    }

    onCalibrationFrameStatus(status) {
      console.log("📊 [CALIBRATION DIALOG] Frame status update:", status);
      console.log("📊 [CALIBRATION DIALOG] Status value:", status.status);

      if (!this.calibrationOverlay || !this.calibrationInstruction) {
        console.error("❌ Calibration overlay elements not found!");
        return;
      }

      this.calibrationOverlay.className = "calibration-overlay active";

      let instructionText = "请将脸部对准框内";

      if (status.status === "no-face") {
        this.calibrationOverlay.classList.add("no-face");
        instructionText = "请将脸部放入框内";
      } else if (
        status.status === "align" ||
        status.status === "too-left" ||
        status.status === "too-right" ||
        status.status === "too-high" ||
        status.status === "too-low"
      ) {
        this.calibrationOverlay.classList.add("align");

        // 根据具体状态给出精确指导
        if (status.status === "too-left") {
          instructionText = "请向右移动";
        } else if (status.status === "too-right") {
          instructionText = "请向左移动";
        } else if (status.status === "too-high") {
          instructionText = "请向下移动";
        } else if (status.status === "too-low") {
          instructionText = "请向上移动";
        } else {
          instructionText = "请调整位置对准框内";
        }
      } else if (status.status === "good") {
        this.calibrationOverlay.classList.add("ready");
        instructionText = "位置完美！可以开始校准";
      } else if (status.status === "too-close") {
        this.calibrationOverlay.classList.add("too-close");
        instructionText = "请远离一些";
      } else if (status.status === "too-far") {
        this.calibrationOverlay.classList.add("too-far");
        instructionText = "请靠近一些";
      }

      this.calibrationInstruction.textContent = instructionText;
      this.updateStatus(this.getStatusType(status.status), instructionText);

      const shouldEnable = status.status === "good";
      this.startCalibrationBtn.disabled = !shouldEnable;

      console.log("🎯 Button state update:", {
        status: status.status,
        shouldEnable: shouldEnable,
        buttonDisabled: this.startCalibrationBtn.disabled,
      });
    }

    getStatusType(status) {
      const types = {
        "no-face": "waiting",
        align: "detecting",
        "too-left": "detecting",
        "too-right": "detecting",
        "too-high": "detecting",
        "too-low": "detecting",
        "too-close": "error",
        "too-far": "error",
        good: "aligned",
      };
      return types[status] || "waiting";
    }

    updateFontPreview() {
      this.fontPreview.style.fontSize = this.baseFontSizeValue + "px";
    }

    async startCalibration() {
      if (!this.distanceDetector) {
        this.showToast("请先启动摄像头", "error");
        return;
      }

      this.isCalibrating = true;
      this.startCalibrationBtn.disabled = true;
      this.updateStatus("detecting", "校准中... 请保持位置不动");

      let countdown = 3;
      const countdownInterval = setInterval(() => {
        this.updateStatus("detecting", `校准中... ${countdown}秒`);
        countdown--;

        if (countdown < 0) {
          clearInterval(countdownInterval);
          this.completeCalibration();
        }
      }, 1000);
    }

    async completeCalibration() {
      try {
        const calibrationResult = await this.distanceDetector.calibrate(
          this.baseFontSizeValue
        );

        this.calibrationData = {
          referenceFaceWidth: calibrationResult.referenceFaceWidth,
          referenceFontSize: calibrationResult.referenceFontSize,
          timestamp: calibrationResult.timestamp,
        };

        this.updateStatus("aligned", "校准完成！请点击保存校准");
        this.saveCalibrationBtn.disabled = false;
        this.showToast("校准完成！", "success");
      } catch (error) {
        console.error("❌ Calibration failed:", error);
        this.updateStatus("error", "校准失败，请重试");
        this.showToast("校准失败，请确保人脸在检测范围内", "error");
      }

      this.isCalibrating = false;
      this.startCalibrationBtn.disabled = false;
    }

    saveCalibration() {
      if (!this.calibrationData) {
        this.showToast("请先完成校准", "error");
        return;
      }

      try {
        localStorage.setItem(
          "mediapipe-calibration",
          JSON.stringify(this.calibrationData)
        );
        this.showToast("校准数据已保存！", "success");
        this.saveCalibrationBtn.disabled = true;
        this.updateStatus("aligned", "校准数据已保存，可以使用动态字体功能");

        // Close dialog after successful save
        setTimeout(() => {
          this.cleanup();
          this.dialog.remove();
        }, 1500);
      } catch (error) {
        console.error("❌ Failed to save calibration:", error);
        this.showToast("保存失败，请重试", "error");
      }
    }

    resetCalibration() {
      try {
        localStorage.removeItem("mediapipe-calibration");
        this.calibrationData = null;
        this.saveCalibrationBtn.disabled = true;
        this.updateStatus("waiting", "校准数据已清除");
        this.showToast("校准数据已清除", "success");

        if (this.distanceDetector) {
          this.distanceDetector.calibration = {
            isCalibrated: false,
            referenceFaceWidth: null,
            referenceDistance: 0,
            timestamp: null,
          };
        }
      } catch (error) {
        console.error("❌ Failed to reset calibration:", error);
        this.showToast("重置失败，请重试", "error");
      }
    }

    checkExistingCalibration() {
      try {
        const saved = localStorage.getItem("mediapipe-calibration");
        if (saved) {
          this.calibrationData = JSON.parse(saved);
          if (this.calibrationData.referenceFontSize) {
            this.baseFontSize.value = this.calibrationData.referenceFontSize;
            this.baseFontValue.textContent =
              this.calibrationData.referenceFontSize + "px";
            this.baseFontSizeValue = this.calibrationData.referenceFontSize;
            this.updateFontPreview();
          }
          this.updateStatus("aligned", "检测到已保存的校准数据");
        }
      } catch (error) {
        console.error("❌ Failed to load calibration:", error);
      }
    }

    updateStatus(type, message) {
      this.statusIndicator.className = `status-indicator ${type}`;
      this.statusText.textContent = message;
    }

    onError(error) {
      console.error("❌ Distance detector error:", error);
      this.updateStatus("error", "检测错误: " + error.message);
      this.showToast("检测错误: " + error.message, "error");
    }

    showToast(message, type = "info") {
      const toast = document.createElement("div");
      toast.className = `calibration-toast ${type}`;
      toast.textContent = message;
      toast.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${
          type === "error" ? "#ff4d4f" : type === "success" ? "#52c41a" : "#333"
        };
        color: white;
        padding: 12px 20px;
        border-radius: 6px;
        z-index: 10003;
        transform: translateX(100%);
        transition: transform 0.3s ease;
      `;
      document.body.appendChild(toast);

      setTimeout(() => {
        toast.style.transform = "translateX(0)";
      }, 100);

      setTimeout(() => {
        toast.style.transform = "translateX(100%)";
        setTimeout(() => {
          if (toast.parentNode) {
            document.body.removeChild(toast);
          }
        }, 300);
      }, 3000);
    }

    cleanup() {
      if (this.distanceDetector) {
        console.log("🧹 Cleaning up calibration dialog detector");
        this.distanceDetector.cleanup();
        this.distanceDetector = null;
      }

      // 额外清理视频元素
      if (this.videoElement && this.videoElement.srcObject) {
        const tracks = this.videoElement.srcObject.getTracks();
        tracks.forEach((track) => track.stop());
        this.videoElement.srcObject = null;
      }

      console.log("✅ Calibration dialog cleanup complete");
    }
  }

  // ===== 校准管理器 =====
  class CalibrationManager {
    static saveCalibration(data) {
      console.log("💾 Saving calibration data:", data);
      localStorage.setItem("mediapipe-calibration", JSON.stringify(data));
    }

    static getCalibration() {
      try {
        const data = localStorage.getItem("mediapipe-calibration");
        const calibrationData = data ? JSON.parse(data) : null;
        console.log("📥 Loading calibration data:", calibrationData);
        return calibrationData;
      } catch (error) {
        console.error("❌ Error loading calibration data:", error);
        return null;
      }
    }

    static isCalibrated() {
      const calibrationData = this.getCalibration();
      const isCalibrated =
        calibrationData && calibrationData.referenceFaceWidth;
      console.log("🎯 Calibration status:", isCalibrated);
      return isCalibrated;
    }

    static clearCalibration() {
      console.log("🗑️ Clearing calibration data");
      localStorage.removeItem("mediapipe-calibration");
    }
  }

  // ===== 样式定义 =====
  const READING_MODE_STYLES = `
/* Reading Mode Styles - Safari-style Implementation */

/* Floating Button */
.reading-mode-float-btn {
  position: fixed;
  right: 20px;
  top: 50%;
  transform: translateY(-50%);
  width: 48px;
  height: 48px;
  background: rgba(255, 255, 255, 0.95);
  border: 1px solid rgba(0, 0, 0, 0.1);
  border-radius: 24px;
  cursor: pointer;
  z-index: 9999;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  transition: all 0.3s ease;
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
}

.reading-mode-float-btn:hover {
  background: rgba(255, 255, 255, 1);
  transform: translateY(-50%) scale(1.05);
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
}

.reading-mode-float-btn.active {
  background: rgba(24, 144, 255, 0.95);
  border-color: rgba(24, 144, 255, 0.3);
}

.reading-mode-float-btn.active .float-btn-icon {
  color: white;
}

.float-btn-icon {
  color: #666;
  transition: color 0.3s ease;
}

.reading-mode-float-btn:hover .float-btn-icon {
  color: #333;
}

/* Floating Menu */
.reading-mode-float-menu {
  position: fixed;
  right: 80px;
  top: 50%;
  transform: translateY(-50%);
  width: 240px;
  background: rgba(255, 255, 255, 0.95);
  border: 1px solid rgba(0, 0, 0, 0.1);
  border-radius: 12px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
  z-index: 9998;
  opacity: 0;
  visibility: hidden;
  transform: translateY(-50%) translateX(10px);
  transition: all 0.3s ease;
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
}

.reading-mode-float-menu.show {
  opacity: 1;
  visibility: visible;
  transform: translateY(-50%) translateX(0);
}

.float-menu-content {
  padding: 8px 0;
}

.menu-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid rgba(0, 0, 0, 0.1);
  margin-bottom: 4px;
}

.menu-title {
  font-size: 14px;
  font-weight: 600;
  color: #333;
}

.menu-close-btn {
  background: none;
  border: none;
  font-size: 18px;
  color: #666;
  cursor: pointer;
  padding: 0;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
  transition: all 0.2s ease;
}

.menu-close-btn:hover {
  background: rgba(0, 0, 0, 0.1);
  color: #333;
}

.menu-item {
  margin: 2px 8px;
}

.menu-item button {
  width: 100%;
  background: none;
  border: none;
  padding: 10px 12px;
  text-align: left;
  cursor: pointer;
  border-radius: 8px;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  gap: 8px;
}

.menu-item button:hover {
  background: rgba(24, 144, 255, 0.1);
}

.menu-icon {
  font-size: 16px;
  width: 20px;
  text-align: center;
}

.menu-text {
  font-size: 14px;
  color: #333;
  flex: 1;
}

.menu-checkbox {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  cursor: pointer;
  border-radius: 8px;
  transition: all 0.2s ease;
}

.menu-checkbox:hover {
  background: rgba(24, 144, 255, 0.1);
}

.menu-checkbox input[type="checkbox"] {
  display: none;
}

.checkmark {
  width: 18px;
  height: 18px;
  border: 2px solid #ddd;
  border-radius: 4px;
  position: relative;
  transition: all 0.2s ease;
}

.menu-checkbox input[type="checkbox"]:checked + .checkmark {
  background: #1890ff;
  border-color: #1890ff;
}

.menu-checkbox input[type="checkbox"]:checked + .checkmark::after {
  content: "";
  position: absolute;
  left: 4px;
  top: 1px;
  width: 6px;
  height: 10px;
  border: solid white;
  border-width: 0 2px 2px 0;
  transform: rotate(45deg);
}

.menu-divider {
  height: 1px;
  background: rgba(0, 0, 0, 0.1);
  margin: 4px 16px;
}

/* Reading Mode Container - Kindle Sepia Theme */
.reading-mode-container {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: #fbf0d9; /* Kindle Sepia background */
  color: #5f4b32; /* Kindle Sepia text color */
  z-index: 10000;
  display: flex;
  flex-direction: column;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
    "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
}

/* Header styling with sepia theme */
.reading-mode-header {
  background: #fbf0d9; /* Same as container background */
  border-bottom: 1px solid rgba(95, 75, 50, 0.2); /* Subtle sepia border */
  padding: 16px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky;
  top: 0;
  z-index: 10001;
}

.reading-mode-close-btn {
  background: none;
  border: none;
  font-size: 24px;
  color: #5f4b32; /* Sepia text color */
  cursor: pointer;
  padding: 4px;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 6px;
  transition: all 0.2s ease;
}

.reading-mode-close-btn:hover {
  background: rgba(95, 75, 50, 0.1); /* Light sepia hover */
  color: #5f4b32; /* Keep same text color */
}

.reading-mode-title {
  font-size: 18px;
  font-weight: 600;
  color: #5f4b32; /* Sepia text color */
  text-align: center;
  flex: 1;
  margin: 0 20px;
}

.reading-mode-content {
  flex: 1;
  overflow-y: auto;
  padding: 40px 20px;
  display: flex;
  justify-content: center;
  background: #fbf0d9; /* Sepia background */
  transition: background-color 0.3s ease, background 0.3s ease;
}

.reading-article-content {
  max-width: 680px;
  width: 100%;
  line-height: 1.8;
  color: #5f4b32; /* Sepia text color */
  font-size: 16px; /* 基础字体大小，会被JavaScript动态覆盖 */
  transition: font-size 0.3s ease, color 0.3s ease;
}

.reading-article-content p {
  margin-bottom: 20px;
  text-align: justify;
  color: #5f4b32; /* Sepia text color */
}

.reading-article-content h1,
.reading-article-content h2,
.reading-article-content h3,
.reading-article-content h4,
.reading-article-content h5,
.reading-article-content h6 {
  margin: 32px 0 16px 0;
  color: #5f4b32; /* Same sepia color for consistency */
  line-height: 1.4;
}

.reading-article-content h1 {
  font-size: 1.8em;
  font-weight: 700;
}

.reading-article-content h2 {
  font-size: 1.5em;
  font-weight: 600;
}

.reading-article-content h3 {
  font-size: 1.3em;
  font-weight: 600;
}

/* All text elements should use sepia color */
.reading-article-content strong,
.reading-article-content b,
.reading-article-content em,
.reading-article-content i,
.reading-article-content a {
  color: #5f4b32; /* Consistent sepia text color */
}

.reading-article-content a:hover {
  color: #5f4b32; /* Keep same color on hover */
  text-decoration: underline;
}

.reading-mode-footer {
  background: #fbf0d9; /* Sepia background */
  border-top: 1px solid rgba(95, 75, 50, 0.2); /* Subtle sepia border */
  padding: 12px 24px;
}

.reading-mode-info {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 12px;
  color: #5f4b32; /* Sepia text color */
}

.reading-mode-info-left {
  display: flex;
  align-items: center;
  gap: 16px;
}

.reading-mode-info-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

.font-size-info,
.word-count-info,
.reading-time-info,
.reading-timer-info {
  font-weight: 500;
  color: #5f4b32; /* Sepia text color */
}

.reading-timer-info {
  background: rgba(95, 75, 50, 0.1);
  padding: 4px 8px;
  border-radius: 4px;
  font-family: "Courier New", monospace;
}

#readingTimer {
  font-weight: 600;
  color: #5f4b32;
}

.dynamic-font-status,
.dynamic-contrast-status {
  color: #5f4b32; /* Same sepia color for consistency */
  font-weight: 500;
}

/* Calibration Dialog */
.calibration-dialog-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 10002;
  backdrop-filter: blur(4px);
  -webkit-backdrop-filter: blur(4px);
}

.calibration-dialog {
  background: white;
  border-radius: 12px;
  padding: 0;
  max-width: 400px;
  width: 90%;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
  overflow: hidden;
}

.dialog-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 20px 24px;
  border-bottom: 1px solid #f0f0f0;
}

.dialog-header h3 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
  color: #333;
}

.dialog-close-btn {
  background: none;
  border: none;
  font-size: 20px;
  color: #666;
  cursor: pointer;
  padding: 0;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
  transition: all 0.2s ease;
}

.dialog-close-btn:hover {
  background: rgba(0, 0, 0, 0.1);
  color: #333;
}

.dialog-content {
  padding: 24px;
  line-height: 1.6;
  color: #666;
}

.dialog-content p {
  margin: 0 0 12px 0;
}

.dialog-actions {
  display: flex;
  gap: 12px;
  padding: 20px 24px;
  border-top: 1px solid #f0f0f0;
  justify-content: flex-end;
}

.btn-primary,
.btn-secondary {
  padding: 8px 16px;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  border: none;
}

.btn-primary {
  background: #1890ff;
  color: white;
}

.btn-primary:hover {
  background: #40a9ff;
}

.btn-secondary {
  background: #f5f5f5;
  color: #666;
}

.btn-secondary:hover {
  background: #e6e6e6;
  color: #333;
}

/* Error Toast */
.error-toast {
  position: fixed;
  top: 20px;
  right: 20px;
  background: #ff4757;
  color: white;
  padding: 12px 20px;
  border-radius: 8px;
  font-size: 14px;
  box-shadow: 0 4px 12px rgba(255, 71, 87, 0.3);
  z-index: 10003;
  animation: slideInRight 0.3s ease;
}

/* Success Toast */
.success-toast {
  position: fixed;
  top: 20px;
  right: 20px;
  background: #2ed573;
  color: white;
  padding: 12px 20px;
  border-radius: 8px;
  font-size: 14px;
  box-shadow: 0 4px 12px rgba(46, 213, 115, 0.3);
  z-index: 10003;
  animation: slideInRight 0.3s ease;
}

@keyframes slideInRight {
  from {
    transform: translateX(100%);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}

/* Responsive Design */
@media (max-width: 768px) {
  .reading-mode-float-btn {
    left: 16px;
    width: 44px;
    height: 44px;
  }

  .reading-mode-float-menu {
    left: 72px;
    width: 200px;
  }

  .reading-mode-header {
    padding: 12px 16px;
  }

  .reading-mode-title {
    font-size: 16px;
    margin: 0 12px;
  }

  .reading-mode-content {
    padding: 24px 16px;
  }

  .reading-article-content {
    max-width: 100%;
  }

  .calibration-dialog {
    width: 95%;
    max-width: 320px;
  }

  .dialog-header,
  .dialog-content,
  .dialog-actions {
    padding: 16px 20px;
  }
}

/* Smooth scrolling */
.reading-mode-content {
  scroll-behavior: smooth;
}

/* Selection styling */
.reading-article-content ::selection {
  background: rgba(24, 144, 255, 0.2);
}

/* Focus styles for accessibility */
.reading-mode-close-btn:focus,
.menu-close-btn:focus,
.dialog-close-btn:focus {
  outline: 2px solid #1890ff;
  outline-offset: 2px;
}

.menu-item button:focus {
  outline: 2px solid #1890ff;
  outline-offset: -2px;
}

/* Complete Calibration Dialog Styles */
.calibration-dialog-full {
  background: white;
  border-radius: 12px;
  max-width: 900px;
  width: 95%;
  max-height: 90vh;
  overflow-y: auto;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
}

.calibration-dialog-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 20px 24px;
  border-bottom: 1px solid #f0f0f0;
  background: #fafafa;
  border-radius: 12px 12px 0 0;
}

.calibration-dialog-header h3 {
  margin: 0;
  font-size: 20px;
  font-weight: 600;
  color: #333;
  display: flex;
  align-items: center;
  gap: 8px;
}

.calibration-close-btn {
  background: none;
  border: none;
  font-size: 24px;
  color: #666;
  cursor: pointer;
  padding: 4px;
  border-radius: 4px;
  transition: all 0.2s;
  line-height: 1;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.calibration-close-btn:hover {
  background: #f0f0f0;
  color: #333;
}

.calibration-dialog-body {
  padding: 24px;
}

.calibration-description {
  text-align: center;
  margin-bottom: 24px;
  color: #666;
  font-size: 14px;
  line-height: 1.5;
}

.calibration-container {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 32px;
  margin-bottom: 24px;
}

.calibration-left .video-container {
  position: relative;
  background: #000;
  border-radius: 12px;
  overflow: hidden;
  aspect-ratio: 4/3;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.calibration-left .video-preview {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.calibration-overlay {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 140px;
  height: 180px;
  border: 3px solid transparent;
  border-radius: 12px;
  opacity: 0;
  transition: all 0.3s ease;
  pointer-events: none;
  z-index: 10;
}

.calibration-overlay.active {
  opacity: 1;
}

.calibration-overlay.no-face {
  border-color: #9e9e9e;
  box-shadow: 0 0 20px rgba(158, 158, 158, 0.5);
}

.calibration-overlay.align {
  border-color: #ff9800;
  box-shadow: 0 0 20px rgba(255, 152, 0, 0.8);
}

.calibration-overlay.ready {
  border-color: #4caf50;
  box-shadow: 0 0 20px rgba(76, 175, 80, 0.8);
}

.calibration-overlay.too-close {
  border-color: #f44336;
  box-shadow: 0 0 20px rgba(244, 67, 54, 0.8);
}

.calibration-overlay.too-far {
  border-color: #2196f3;
  box-shadow: 0 0 20px rgba(33, 150, 243, 0.8);
}

.calibration-instruction {
  position: absolute;
  top: -50px;
  left: 50%;
  transform: translateX(-50%);
  color: #fff;
  font-weight: 600;
  font-size: 14px;
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
  white-space: nowrap;
  text-align: center;
  background: rgba(0, 0, 0, 0.8);
  padding: 8px 16px;
  border-radius: 20px;
  backdrop-filter: blur(4px);
}

.calibration-right {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  border: 1px solid;
}

.status-indicator.waiting {
  background: #f0f0f0;
  color: #666;
  border-color: #e0e0e0;
}

.status-indicator.detecting {
  background: #e6f7ff;
  color: #1890ff;
  border-color: #91d5ff;
}

.status-indicator.aligned {
  background: #f6ffed;
  color: #52c41a;
  border-color: #b7eb8f;
}

.status-indicator.error {
  background: #fff2f0;
  color: #ff4d4f;
  border-color: #ffccc7;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: currentColor;
  flex-shrink: 0;
}

.control-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.control-label {
  font-size: 14px;
  font-weight: 600;
  color: #333;
}

.control-hint {
  font-size: 12px;
  color: #666;
  line-height: 1.4;
  margin-top: 4px;
}

.slider-container {
  display: flex;
  align-items: center;
  gap: 12px;
}

.slider {
  flex: 1;
  height: 6px;
  background: #f0f0f0;
  border-radius: 3px;
  outline: none;
  -webkit-appearance: none;
  appearance: none;
}

.slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 20px;
  height: 20px;
  background: #1890ff;
  border-radius: 50%;
  cursor: pointer;
  border: 2px solid white;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

.slider::-moz-range-thumb {
  width: 20px;
  height: 20px;
  background: #1890ff;
  border-radius: 50%;
  cursor: pointer;
  border: 2px solid white;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

.slider-value {
  font-size: 14px;
  color: #1890ff;
  font-weight: 600;
  min-width: 45px;
  text-align: center;
}

.font-preview {
  font-size: 16px;
  padding: 16px;
  border: 2px dashed #e0e0e0;
  border-radius: 8px;
  background: #fafafa;
  text-align: center;
  color: #333;
  font-family: inherit;
  transition: font-size 0.3s ease;
}

.calibration-buttons {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin-top: 8px;
}

.calibration-buttons .btn {
  padding: 10px 16px;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  text-align: center;
  border: 1px solid transparent;
}

.calibration-buttons .btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.calibration-buttons .btn-primary {
  background: #1890ff;
  color: white;
  border-color: #1890ff;
}

.calibration-buttons .btn-primary:hover:not(:disabled) {
  background: #40a9ff;
  border-color: #40a9ff;
}

.calibration-buttons .btn-secondary {
  background: #f0f0f0;
  color: #666;
  border-color: #d9d9d9;
}

.calibration-buttons .btn-secondary:hover:not(:disabled) {
  background: #e0e0e0;
  color: #333;
}

.calibration-buttons .btn-danger {
  background: #ff4d4f;
  color: white;
  border-color: #ff4d4f;
}

.calibration-buttons .btn-danger:hover:not(:disabled) {
  background: #ff7875;
  border-color: #ff7875;
}

.calibration-info {
  background: linear-gradient(135deg, #f0f8ff 0%, #e6f7ff 100%);
  border: 1px solid #91d5ff;
  border-radius: 12px;
  padding: 20px;
  margin-top: 8px;
}

.calibration-info h4 {
  color: #1890ff;
  margin: 0 0 12px 0;
  font-size: 16px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 8px;
}

.calibration-info ul {
  color: #666;
  font-size: 14px;
  margin: 0;
  padding-left: 20px;
  line-height: 1.6;
}

.calibration-info li {
  margin-bottom: 4px;
}

/* Responsive design for calibration dialog */
@media (max-width: 768px) {
  .calibration-dialog-full {
    width: 98%;
    max-height: 95vh;
  }
  
  .calibration-container {
    grid-template-columns: 1fr;
    gap: 20px;
  }
  
  .calibration-dialog-body {
    padding: 16px;
  }
  
  .calibration-buttons {
    grid-template-columns: 1fr;
  }
  
  .calibration-overlay {
    width: 120px;
    height: 160px;
  }
  
  .calibration-instruction {
    font-size: 12px;
    top: -40px;
  }
}

@media (max-width: 480px) {
  .calibration-dialog-header {
    padding: 16px;
  }
  
  .calibration-dialog-header h3 {
    font-size: 18px;
  }
  
  .calibration-overlay {
    width: 100px;
    height: 140px;
  }
}
    `;

  // ===== 主程序入口 =====
  function initializeReadingMode() {
    // 加载样式
    const styleElement = document.createElement("style");
    styleElement.textContent = READING_MODE_STYLES;
    document.head.appendChild(styleElement);
    console.log("✅ 样式已加载");

    // 检查页面是否有可读内容
    if (
      document.querySelector(
        ".content, main, article, [class*='content'], [class*='article'], [class*='post']"
      )
    ) {
      window.readingModeManager = new ReadingModeManager();
      console.log("✅ 完整动态阅读模式已初始化");
    } else {
      console.log("📖 当前页面似乎没有可读内容，跳过初始化");
    }
  }

  // 在DOM加载完成后启动
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initializeReadingMode);
  } else {
    // 小延迟确保页面完全加载
    setTimeout(initializeReadingMode, 500);
  }

  console.log("🚀 完整动态阅读模式用户脚本已加载");
})();
