<template>
    <div class="home">
      <h1>Image Classification</h1>
      <p>Upload an image to identify what's in it using our machine learning model.</p>
      
      <div class="upload-container">
        <div 
          class="upload-area" 
          :class="{ 'active': isDragging }"
          @dragover.prevent="isDragging = true"
          @dragleave.prevent="isDragging = false"
          @drop.prevent="onDrop"
          @click="triggerFileInput"
        >
          <div v-if="!selectedImage">
            <i class="upload-icon">📁</i>
            <p>Drop your image here or click to browse</p>
          </div>
          <div v-else class="preview-container">
            <img :src="selectedImage" alt="Preview" class="image-preview" />
            <button class="remove-btn" @click.stop="removeImage">×</button>
          </div>
          <input 
            type="file" 
            ref="fileInput" 
            @change="onFileSelected" 
            accept="image/*"
            style="display: none"
          />
        </div>
        
        <button 
          class="analyze-btn"
          :disabled="!selectedImage || isAnalyzing"
          @click="analyzeImage"
        >
          {{ isAnalyzing ? 'Analyzing...' : 'Analyze Image' }}
        </button>
      </div>
      
      <div class="results" v-if="results">
        <h2>Results</h2>
        <div class="result-card">
          <div class="result-image">
            <img :src="selectedImage" alt="Analyzed image" />
          </div>
          <div class="result-details">
            <h3>Prediction: {{ results.class }}</h3>
            <div class="accuracy-bar">
              <div class="accuracy-fill" :style="{ width: `${results.accuracy}%` }"></div>
              <span>{{ results.accuracy.toFixed(1) }}% confidence</span>
            </div>
            <p class="timestamp">Analyzed on: {{ results.timestamp }}</p>
          </div>
        </div>
      </div>
    </div>
  </template>
  
  <script>
  import { ref, reactive } from 'vue'
  import { useStore } from 'vuex'
  
  export default {
    setup() {
      const store = useStore()
      const fileInput = ref(null)
      const selectedImage = ref(null)
      const isDragging = ref(false)
      const isAnalyzing = ref(false)
      const results = ref(null)
      const fileObj = ref(null)
      
      const triggerFileInput = () => {
        fileInput.value.click()
      }
      
      const onFileSelected = (event) => {
        const file = event.target.files[0]
        if (file) {
          handleFile(file)
        }
      }
      
      const onDrop = (event) => {
        isDragging.value = false
        const file = event.dataTransfer.files[0]
        if (file && file.type.startsWith('image/')) {
          handleFile(file)
        }
      }
      
      const handleFile = (file) => {
        fileObj.value = file
        const reader = new FileReader()
        reader.onload = (e) => {
          selectedImage.value = e.target.result
        }
        reader.readAsDataURL(file)
        results.value = null
      }
      
      const removeImage = () => {
        selectedImage.value = null
        fileObj.value = null
        results.value = null
      }
      
      const analyzeImage = () => {
        if (!selectedImage.value) return
        
        isAnalyzing.value = true
        
        // Simulate API call to ML model
        setTimeout(() => {
          // Mock response - in a real app this would come from your backend
          const mockResults = {
            class: ['Dog', 'Cat', 'Bird', 'Car', 'Landscape'][Math.floor(Math.random() * 5)],
            accuracy: 70 + Math.random() * 29.9, // Random between 70-99.9%
            timestamp: new Date().toLocaleString(),
            imageUrl: selectedImage.value
          }
          
          results.value = mockResults
          
          // Save to history
          store.dispatch('saveInference', {
            id: Date.now(),
            ...mockResults
          })
          
          isAnalyzing.value = false
        }, 2000)
      }
      
      return {
        fileInput,
        selectedImage,
        isDragging,
        isAnalyzing,
        results,
        triggerFileInput,
        onFileSelected,
        onDrop,
        removeImage,
        analyzeImage
      }
    }
  }
  </script>
  
  <style scoped>
  .home {
    max-width: 800px;
    margin: 0 auto;
  }
  
  .upload-container {
    margin: 2rem 0;
  }
  
  .upload-area {
    border: 2px dashed #ccc;
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
  }
  
  .upload-area.active {
    border-color: #42b983;
    background-color: rgba(66, 185, 131, 0.1);
  }
  
  .upload-icon {
    font-size: 3rem;
    display: block;
    margin-bottom: 1rem;
  }
  
  .preview-container {
    position: relative;
    display: inline-block;
  }
  
  .image-preview {
    max-width: 100%;
    max-height: 300px;
    border-radius: 4px;
  }
  
  .remove-btn {
    position: absolute;
    top: -10px;
    right: -10px;
    width: 25px;
    height: 25px;
    border-radius: 50%;
    background: #ff4757;
    color: white;
    border: none;
    cursor: pointer;
    font-size: 1rem;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  
  .analyze-btn {
    display: block;
    width: 100%;
    padding: 1rem;
    background-color: #42b983;
    color: white;
    border: none;
    border-radius: 4px;
    font-size: 1rem;
    cursor: pointer;
    margin-top: 1rem;
    transition: background-color 0.3s;
  }
  
  .analyze-btn:hover:not(:disabled) {
    background-color: #3aa876;
  }
  
  .analyze-btn:disabled {
    background-color: #ccc;
    cursor: not-allowed;
  }
  
  .results {
    margin-top: 2rem;
  }
  
  .result-card {
    display: flex;
    background-color: #f8f9fa;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  }
  
  .result-image {
    flex: 0 0 40%;
  }
  
  .result-image img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
  
  .result-details {
    flex: 1;
    padding: 1.5rem;
  }
  
  .accuracy-bar {
    margin: 1rem 0;
    background-color: #eee;
    height: 24px;
    border-radius: 12px;
    position: relative;
    overflow: hidden;
  }
  
  .accuracy-fill {
    height: 100%;
    background-color: #42b983;
    border-radius: 12px 0 0 12px;
  }
  
  .accuracy-bar span {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #333;
    font-weight: bold;
  }
  
  .timestamp {
    color: #666;
    font-size: 0.9rem;
  }
  </style>