<template>
    <div class="history">
      <h1>Your Image History</h1>
      <p v-if="history.length === 0" class="empty-history">
        You haven't analyzed any images yet. Go to the home page to get started.
      </p>
      
      <div class="history-list" v-else>
        <div 
          v-for="item in history" 
          :key="item.id" 
          class="history-item"
        >
          <div class="history-image">
            <img :src="item.imageUrl" alt="History image" />
          </div>
          <div class="history-details">
            <h3>{{ item.class }}</h3>
            <div class="accuracy-bar">
              <div class="accuracy-fill" :style="{ width: `${item.accuracy}%` }"></div>
              <span>{{ item.accuracy.toFixed(1) }}% confidence</span>
            </div>
            <p class="timestamp">{{ item.timestamp }}</p>
            <button class="download-btn" @click="downloadImage(item)">
              Download Image
            </button>
          </div>
        </div>
      </div>
    </div>
  </template>
  
  <script>
  import { computed } from 'vue'
  import { useStore } from 'vuex'
  
  export default {
    setup() {
      const store = useStore()
      
      const history = computed(() => store.state.history)
      
      const downloadImage = (item) => {
        // Create a temporary anchor element
        const link = document.createElement('a')
        link.href = item.imageUrl
        link.download = `ml-vision-${item.class}-${Date.now()}.jpg`
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
      }
      
      return {
        history,
        downloadImage
      }
    }
  }
  </script>
  
  <style scoped>
  .history {
    max-width: 800px;
    margin: 0 auto;
  }
  
  .empty-history {
    text-align: center;
    padding: 3rem 0;
    color: #666;
  }
  
  .history-list {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
    margin-top: 2rem;
  }
  
  .history-item {
    display: flex;
    background-color: #f8f9fa;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }
  
  .history-image {
    flex: 0 0 180px;
  }
  
  .history-image img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
  
  .history-details {
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
    margin-bottom: 1rem;
  }
  
  .download-btn {
    padding: 0.5rem 1rem;
    background-color: #3498db;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    transition: background-color 0.3s;
  }
  
  .download-btn:hover {
    background-color: #2980b9;
  }
  
  @media (max-width: 600px) {
    .history-item {
      flex-direction: column;
    }
    
    .history-image {
      flex: 0 0 auto;
      height: 200px;
    }
  }
  </style>