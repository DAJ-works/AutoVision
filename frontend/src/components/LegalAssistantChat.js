// Frontend component for chatting with the CA Vehicle Code expert
// Available anytime - not just after case analysis

import React, { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import { 
  Box, Typography, Paper, CircularProgress, IconButton,
  TextField, Button, Divider, Tooltip, Avatar, Chip, Alert
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import SendIcon from '@mui/icons-material/Send';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import PersonIcon from '@mui/icons-material/Person';
import GavelIcon from '@mui/icons-material/Gavel';
import InfoIcon from '@mui/icons-material/Info';

const LegalAssistantChat = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Initialize with welcome message
  useEffect(() => {
    setMessages([
      {
        role: 'assistant',
        content: `Hello! I'm your California Vehicle Code legal assistant, trained on the entire California Vehicle Code. 

I can help you with:
• Traffic law questions (e.g., "What is CVC §22350?")
• Fault determination in accidents
• Traffic violations and penalties
• Vehicle operation regulations
• Insurance and reporting requirements

Ask me anything about California traffic laws!`
      }
    ]);
    
    // Focus the input field
    setTimeout(() => {
      inputRef.current?.focus();
    }, 500);
  }, []);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Add user message to chat
    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    
    // Clear input and show loading state
    setInput('');
    setIsLoading(true);

    try {
      // Send message to backend (general law chat endpoint)
      const response = await axios.post('/api/legal-chat', {
        message: input
      });

      // Add response to chat
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: response.data.response 
      }]);
    } catch (error) {
      console.error('Error getting chat response:', error);
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error processing your request. Please try again.' 
      }]);
    } finally {
      setIsLoading(false);
      // Focus input again after response
      inputRef.current?.focus();
    }
  };

  // Suggested questions for quick start
  const suggestedQuestions = [
    "What is the basic speed law in California?",
    "Who is at fault if someone runs a red light?",
    "What are the accident reporting requirements?",
    "Explain right-of-way rules at intersections",
    "What is reckless driving under California law?"
  ];

  const handleSuggestedQuestion = (question) => {
    setInput(question);
    inputRef.current?.focus();
  };

  return (
    <Paper sx={{ 
      height: 'calc(100vh - 100px)', 
      display: 'flex', 
      flexDirection: 'column',
      borderRadius: 2,
      overflow: 'hidden'
    }}>
      {/* Chat Header */}
      <Box sx={{ 
        p: 2, 
        background: 'linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%)',
        color: 'white',
        display: 'flex',
        alignItems: 'center',
        gap: 2
      }}>
        <Avatar sx={{ bgcolor: 'rgba(255,255,255,0.2)', width: 48, height: 48 }}>
          <GavelIcon sx={{ fontSize: 28 }} />
        </Avatar>
        <Box sx={{ flex: 1 }}>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            CA Vehicle Code Legal Assistant
          </Typography>
          <Typography variant="body2" sx={{ opacity: 0.9 }}>
            Trained on the complete California Vehicle Code
          </Typography>
        </Box>
        <Tooltip title="Back to Dashboard">
          <IconButton 
            component={Link} 
            to="/dashboard" 
            sx={{ color: 'white' }}
            size="small"
          >
            <ArrowBackIcon />
          </IconButton>
        </Tooltip>
      </Box>

      {/* Info Banner */}
      <Alert 
        severity="info" 
        icon={<InfoIcon />}
        sx={{ 
          borderRadius: 0,
          '& .MuiAlert-message': { width: '100%' }
        }}
      >
        <Typography variant="body2">
          This AI assistant has been fine-tuned on California Vehicle Code and can answer any traffic law question instantly. 
          It can also analyze accident cases when you provide details.
        </Typography>
      </Alert>
      
      {/* Chat Messages */}
      <Box sx={{ 
        flexGrow: 1, 
        overflowY: 'auto',
        p: 3,
        display: 'flex',
        flexDirection: 'column',
        gap: 2,
        bgcolor: theme => theme.palette.mode === 'light' ? '#f5f7fb' : '#111927'
      }}>
        {messages.map((message, index) => (
          <Box 
            key={index} 
            sx={{ 
              display: 'flex',
              flexDirection: 'row',
              alignItems: 'flex-start',
              alignSelf: message.role === 'user' ? 'flex-end' : 'flex-start',
              maxWidth: '85%'
            }}
          >
            {message.role === 'assistant' && (
              <Avatar 
                sx={{ 
                  bgcolor: 'primary.main', 
                  mr: 1.5,
                  width: 36,
                  height: 36
                }}
              >
                <GavelIcon sx={{ fontSize: 20 }} />
              </Avatar>
            )}
            
            <Paper 
              elevation={1}
              sx={{ 
                p: 2,
                borderRadius: 2,
                borderBottomRightRadius: message.role === 'user' ? 0 : 2,
                borderBottomLeftRadius: message.role === 'user' ? 2 : 0,
                bgcolor: message.role === 'user' ? 'primary.main' : 'background.paper',
                color: message.role === 'user' ? 'primary.contrastText' : 'text.primary',
              }}
            >
              <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                {message.content}
              </Typography>
            </Paper>
            
            {message.role === 'user' && (
              <Avatar 
                sx={{ 
                  bgcolor: 'secondary.main', 
                  ml: 1.5,
                  width: 36,
                  height: 36
                }}
              >
                <PersonIcon sx={{ fontSize: 20 }} />
              </Avatar>
            )}
          </Box>
        ))}
        
        {/* Suggested Questions (show only at start) */}
        {messages.length === 1 && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
              Try asking:
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {suggestedQuestions.map((question, idx) => (
                <Chip
                  key={idx}
                  label={question}
                  onClick={() => handleSuggestedQuestion(question)}
                  sx={{ cursor: 'pointer' }}
                  variant="outlined"
                />
              ))}
            </Box>
          </Box>
        )}
        
        {isLoading && (
          <Box 
            sx={{ 
              display: 'flex',
              flexDirection: 'row',
              alignItems: 'flex-start',
              alignSelf: 'flex-start',
              maxWidth: '80%'
            }}
          >
            <Avatar 
              sx={{ 
                bgcolor: 'primary.main', 
                mr: 1.5,
                width: 36,
                height: 36
              }}
            >
              <GavelIcon sx={{ fontSize: 20 }} />
            </Avatar>
            
            <Paper 
              elevation={1}
              sx={{ 
                p: 2,
                borderRadius: 2,
                borderBottomLeftRadius: 0,
                minWidth: 60,
                display: 'flex',
                justifyContent: 'center'
              }}
            >
              <CircularProgress size={20} thickness={5} />
            </Paper>
          </Box>
        )}
        
        <Box ref={messagesEndRef} />
      </Box>
      
      {/* Chat Input */}
      <Box 
        component="form" 
        onSubmit={handleSubmit}
        sx={{ 
          p: 2, 
          bgcolor: 'background.paper',
          borderTop: theme => `1px solid ${theme.palette.divider}`,
          display: 'flex',
          gap: 1
        }}
      >
        <TextField
          fullWidth
          placeholder="Ask about California traffic laws or accident analysis..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          variant="outlined"
          size="medium"
          disabled={isLoading}
          inputRef={inputRef}
          multiline
          maxRows={4}
          sx={{ 
            '& .MuiOutlinedInput-root': {
              borderRadius: 3,
              bgcolor: theme => theme.palette.mode === 'light' ? '#f5f7fb' : '#1a2436'
            }
          }}
        />
        <Button
          type="submit"
          variant="contained"
          color="primary"
          disabled={isLoading || !input.trim()}
          endIcon={<SendIcon />}
          sx={{ borderRadius: 2, px: 3, minWidth: 100 }}
        >
          Send
        </Button>
      </Box>
    </Paper>
  );
};

export default LegalAssistantChat;
