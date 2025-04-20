import { createStore } from 'vuex'

export default createStore({
  state: {
    isAuthenticated: false,
    user: null,
    history: []
  },
  mutations: {
    setAuth(state, status) {
      state.isAuthenticated = status
    },
    setUser(state, user) {
      state.user = user
    },
    addToHistory(state, item) {
      state.history.unshift(item)
    },
    setHistory(state, history) {
      state.history = history
    }
  },
  actions: {
    login({ commit }, credentials) {
      // In a real app, this would make an API call
      return new Promise((resolve) => {
        setTimeout(() => {
          commit('setAuth', true)
          commit('setUser', {
            id: 1,
            username: credentials.username,
            email: `${credentials.username}@example.com`
          })
          
          // Load history from localStorage or initialize it
          const savedHistory = localStorage.getItem(`user_${credentials.username}_history`)
          if (savedHistory) {
            commit('setHistory', JSON.parse(savedHistory))
          } else {
            commit('setHistory', [])
          }
          
          resolve(true)
        }, 1000)
      })
    },
    logout({ commit, state }) {
      // Save history to localStorage before logout
      if (state.user) {
        localStorage.setItem(`user_${state.user.username}_history`, JSON.stringify(state.history))
      }
      
      commit('setAuth', false)
      commit('setUser', null)
      commit('setHistory', [])
    },
    saveInference({ commit, state }, inference) {
      commit('addToHistory', inference)
      
      // Save updated history to localStorage
      if (state.user) {
        localStorage.setItem(`user_${state.user.username}_history`, JSON.stringify(state.history))
      }
    }
  }
})