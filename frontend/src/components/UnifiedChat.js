// Unified AI Chat Component
// Can answer BOTH general CA Vehicle Code questions AND case-specific analysis
// Automatically detects if viewing a case or general assistant

import React, { useState, useEffect, useRef } from 'react';
import { useParams, Link, useLocation } from 'react-router-dom';
import axios from 'axios';
import { 
  Box, Typography, Paper, CircularProgress, IconButton,
  TextField, Button, Divider, Tooltip, Avatar, Chip, Alert,
  Select, MenuItem, FormControl, Badge, InputLabel
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import SendIcon from '@mui/icons-material/Send';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import PersonIcon from '@mui/icons-material/Person';
import GavelIcon from '@mui/icons-material/Gavel';
import InfoIcon from '@mui/icons-material/Info';
import VideoLibraryIcon from '@mui/icons-material/VideoLibrary';
import MenuBookIcon from '@mui/icons-material/MenuBook';

const UnifiedChat = ({ mode = 'auto' }) => {
  // Auto-detect mode based on route if not specified
  const { caseId: routeCaseId } = useParams();
  const location = useLocation();
  
  // State for case selection
  const [selectedCaseId, setSelectedCaseId] = useState(routeCaseId || null);
  const [availableCases, setAvailableCases] = useState([]);
  
  // Determine chat mode: 'case' for case-specific, 'general' for legal assistant
  const [chatMode, setChatMode] = useState(
    mode === 'auto' ? (routeCaseId ? 'case' : 'general') : mode
  );
  
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [caseInfo, setCaseInfo] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Fetch available cases for the selector
  useEffect(() => {
    const fetchCases = async () => {
      try {
        const response = await axios.get('/api/cases');
        const completedCases = response.data.filter(c => c.status === 'completed');
        setAvailableCases(completedCases);
      } catch (error) {
        console.error('Error fetching cases:', error);
      }
    };
    
    fetchCases();
  }, []);

  // Fetch case info when selected case changes
  useEffect(() => {
    const fetchCaseInfo = async () => {
      if (selectedCaseId) {
        try {
          const response = await axios.get(`/api/cases/${selectedCaseId}`);
          setCaseInfo(response.data);
          setChatMode('case');
          
          // Update messages with case context
          setMessages([
            {
              role: 'assistant',
              content: `Hello! I'm your AI legal assistant with expertise in California Vehicle Code.

I can see you're viewing **${response.data.case_name || `Case #${selectedCaseId}`}**. I can help you with:

🔍 **Case Analysis**: Ask about specific incidents, violations, or persons detected in this video
📖 **Legal Questions**: Any question about California traffic laws, fault determination, or penalties

You can switch between case-specific analysis and general legal questions anytime. What would you like to know?`
            }
          ]);
        } catch (error) {
          console.error('Error fetching case info:', error);
          // Fall back to general mode
          setChatMode('general');
        }
      } else {
        // General legal assistant mode
        setCaseInfo(null);
        setChatMode('general');
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
• Analysis of video evidence and incidents

Select a case above to analyze specific incidents, or ask me anything about California traffic laws!`
          }
        ]);
      }
      
      // Focus the input field
      setTimeout(() => {
        inputRef.current?.focus();
      }, 500);
    };

    fetchCaseInfo();
  }, [selectedCaseId]);

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
    const currentInput = input;
    setInput('');
    setIsLoading(true);

    try {
      let response;
      
      // Choose endpoint based on mode and selected case
      if (chatMode === 'case' && selectedCaseId) {
        // Case-specific chat with legal expertise
        response = await axios.post('/api/chat-enhanced', {
          caseId: selectedCaseId,
          message: currentInput
        });
      } else {
        // General legal assistant chat
        response = await axios.post('/api/legal-chat', {
          message: currentInput
        });
      }

      // Add response to chat
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: response.data.response 
      }]);
    } catch (error) {
      console.error('Error getting chat response:', error);
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error processing your request. Please try again or rephrase your question.' 
      }]);
    } finally {
      setIsLoading(false);
      // Focus input again after response
      inputRef.current?.focus();
    }
  };

  // Suggested questions
  const generalQuestions = [
    "What is the basic speed law in California?",
    "Who is at fault if someone runs a red light?",
    "What are the accident reporting requirements?",
    "Explain right-of-way rules at intersections",
    "What is reckless driving under California law?"
  ];

  const caseQuestions = [
    "Summarize the key findings from this video",
    "What violations were detected?",
    "Who might be at fault based on the video?",
    "What California Vehicle Code sections apply?",
    "What are the legal implications?"
  ];

  const suggestedQuestions = (chatMode === 'case' && (selectedCaseId || routeCaseId)) ? caseQuestions : generalQuestions;

  const handleSuggestedQuestion = (question) => {
    setInput(question);
    inputRef.current?.focus();
  };

  // Toggle between modes
  const toggleMode = () => {
    if (selectedCaseId || routeCaseId) {
      const newMode = chatMode === 'case' ? 'general' : 'case';
      setChatMode(newMode);
      
      // Add a system message about mode change
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: newMode === 'case' 
          ? `Switched to **Case Analysis Mode**. I'll now focus on ${caseInfo?.case_name || `Case #${selectedCaseId || routeCaseId}`} while still answering legal questions.`
          : `Switched to **General Legal Mode**. I'll focus on California Vehicle Code questions without specific case context.`
      }]);
    }
  };

  // Get header info based on mode
  const getHeaderInfo = () => {
    if (chatMode === 'case' && (selectedCaseId || routeCaseId)) {
      return {
        title: 'AI Legal Assistant',
        subtitle: `${caseInfo?.case_name || `Case #${selectedCaseId || routeCaseId}`} • CA Vehicle Code Expert`,
        icon: <Badge badgeContent="Case" color="secondary"><GavelIcon sx={{ fontSize: 28 }} /></Badge>,
        gradient: 'linear-gradient(135deg, #7c3aed 0%, #a855f7 100%)'
      };
    }
    return {
      title: 'CA Vehicle Code Legal Assistant',
      subtitle: 'Trained on the complete California Vehicle Code',
      icon: <GavelIcon sx={{ fontSize: 28 }} />,
      gradient: 'linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%)'
    };
  };

  const headerInfo = getHeaderInfo();

  return (
    <Box sx={{ px: { xs: 3, sm: 5, md: 8 }, maxWidth: '1400px', mx: 'auto' }}>
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
        background: headerInfo.gradient,
        color: 'white',
        display: 'flex',
        alignItems: 'center',
        gap: 2
      }}>
        <Avatar sx={{ bgcolor: 'rgba(255,255,255,0.2)', width: 48, height: 48 }}>
          {headerInfo.icon}
        </Avatar>
        <Box sx={{ flex: 1 }}>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            {headerInfo.title}
          </Typography>
          <Typography variant="body2" sx={{ opacity: 0.9 }}>
            {headerInfo.subtitle}
          </Typography>
        </Box>
        
        {/* Mode Toggle (only show if a case is selected) */}
        {(selectedCaseId || routeCaseId) && (
          <Tooltip title={`Switch to ${chatMode === 'case' ? 'General' : 'Case'} Mode`}>
            <IconButton 
              onClick={toggleMode}
              sx={{ 
                color: 'white',
                bgcolor: 'rgba(255,255,255,0.1)',
                '&:hover': { bgcolor: 'rgba(255,255,255,0.2)' }
              }}
              size="small"
            >
              {chatMode === 'case' ? <MenuBookIcon /> : <VideoLibraryIcon />}
            </IconButton>
          </Tooltip>
        )}
        
        <Tooltip title={routeCaseId ? 'Back to Case' : 'Back to Dashboard'}>
          <IconButton 
            component={Link} 
            to={routeCaseId ? `/incidents/${routeCaseId}` : '/dashboard'}
            sx={{ color: 'white' }}
            size="small"
          >
            <ArrowBackIcon />
          </IconButton>
        </Tooltip>
      </Box>

      {/* Case Selector */}
      {!routeCaseId && availableCases.length > 0 && (
        <Box sx={{ 
          p: 2, 
          bgcolor: 'background.paper',
          borderBottom: theme => `1px solid ${theme.palette.divider}`,
          display: 'flex',
          alignItems: 'center',
          gap: 2
        }}>
          <VideoLibraryIcon color="primary" />
          <Typography variant="body2" sx={{ fontWeight: 600 }}>
            Analyze Case:
          </Typography>
          <FormControl size="small" sx={{ flex: 1, maxWidth: 400 }}>
            <Select
              value={selectedCaseId || ''}
              onChange={(e) => setSelectedCaseId(e.target.value || null)}
              displayEmpty
              sx={{ 
                bgcolor: theme => theme.palette.mode === 'light' ? '#f5f7fb' : '#1a2436',
                '& .MuiSelect-select': {
                  py: 1
                }
              }}
            >
              <MenuItem value="">
                <em>General Legal Questions (No Case)</em>
              </MenuItem>
              {availableCases.map((c) => (
                <MenuItem key={c.case_id} value={c.case_id}>
                  {c.case_name || c.case_id} • {c.num_vehicles || 0} vehicles detected
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          {selectedCaseId && (
            <Chip 
              label="Case Mode" 
              color="secondary" 
              size="small"
              onDelete={() => setSelectedCaseId(null)}
            />
          )}
        </Box>
      )}

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
          {chatMode === 'case' && selectedCaseId
            ? `Analyzing ${caseInfo?.case_name || 'this case'} with CA Vehicle Code expertise. I can answer both case-specific and general legal questions.`
            : 'This AI assistant is trained on California Vehicle Code and can answer any traffic law question instantly. Select a case above to analyze specific incidents.'
          }
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
                  bgcolor: chatMode === 'case' ? '#7c3aed' : 'primary.main',
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
              <Typography 
                variant="body1" 
                sx={{ 
                  whiteSpace: 'pre-wrap',
                  '& strong': { fontWeight: 600, color: 'primary.main' }
                }}
                dangerouslySetInnerHTML={{
                  __html: message.content
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\n/g, '<br />')
                }}
              />
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
                  color={chatMode === 'case' ? 'secondary' : 'primary'}
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
                bgcolor: chatMode === 'case' ? '#7c3aed' : 'primary.main',
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
          placeholder={
            chatMode === 'case' && (selectedCaseId || routeCaseId)
              ? `Ask about ${caseInfo?.case_name || 'this case'} or any CA traffic law question...`
              : "Ask about California traffic laws or select a case to analyze..."
          }
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
    </Box>
  );
};

export default UnifiedChat;
