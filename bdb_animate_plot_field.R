# Animate/Plot Play - Based on work from https://www.kaggle.com/adamsonty/nfl-big-data-bowl-a-basic-field-control-model

bdb_animate_plot_field <- function(ex_game, ex_play, ex_track, ex_play_control, method){
  
  # function to plot a general football field
  plot_field <- plot_field()
  
  # extract los, first down line, and team colors (changed JAX black to white for readability)
  line_of_scrimmage = ex_play$line_of_scrimmage
  play_title = paste(ex_game$visitorTeamAbbr, ' @ ', ex_game$homeTeamAbbr, ', ',
                     substr(ex_game$gameId, 1, 4),'-',substr(ex_game$gameId,5,6),
                     '-',substr(ex_game$gameId,7,8),sep='')
  
  # allows for snapshot at the snap, total play animation, and field control animation
  if(method == 'animate'){
    play_frames <- plot_field + 
      # line of scrimmage
      annotate(
        "segment",
        x = line_of_scrimmage, xend = line_of_scrimmage, y = 0, yend = 160/3,
        colour = "#0d41e1", size = 1.5
      ) +
     # away team velocities
      geom_segment(
        data = ex_track %>% filter(team == ex_game$visitorTeamAbbr),
        mapping = aes(x = x, y = y, xend = x + v_x, yend = y + v_y),
        colour = ex_game$away_1, size = 1, arrow = arrow(length = unit(0.01, "npc"))
      ) + 
      # home team velocities
      geom_segment(
        data = ex_track %>% filter(team == ex_game$homeTeamAbbr),
        mapping = aes(x = x, y = y, xend = x + v_x, yend = y + v_y),
        colour = ex_game$home_2, size = 1, arrow = arrow(length = unit(0.01, "npc"))
      ) +
      # away team locs and jersey numbers
      geom_point(
        data = ex_track %>% filter(team == ex_game$visitorTeamAbbr),
        mapping = aes(x = x, y = y),
        fill = "#f8f9fa", colour = ex_game$away_2,
        shape = 21, alpha = 1, size = 8, stroke = 1.5
      ) +
      geom_text(
        data = ex_track %>% filter(team == ex_game$visitorTeamAbbr),
        mapping = aes(x = x, y = y, label = jerseyNumber),
        colour = ex_game$away_1, size = 4.5
      ) +
      # home team locs and jersey numbers
      geom_point(
        data = ex_track %>% filter(team == ex_game$homeTeamAbbr),
        mapping = aes(x = x, y = y),
        fill = ex_game$home_1, colour = ex_game$home_2,
        shape = 21, alpha = 1, size = 8, stroke = 1.5
      ) +
      geom_text(
        data = ex_track %>% filter(team == ex_game$homeTeamAbbr),
        mapping = aes(x = x, y = y, label = jerseyNumber),
        colour = ex_game$home_2, size = 4.5, 
      ) +
      # ball
      geom_point(
        data = ex_track %>% filter(team == "football"),
        mapping = aes(x = x, y = y),
        fill = "#935e38", colour = "#d9d9d9",
        shape = 21, alpha = 1, size = 4, stroke = 1
      ) +
      # title 
      labs(title = play_title,
           subtitle = ex_play$playDescription) +
      theme(plot.title = element_text(face = "bold"),
            plot.subtitle = element_text(hjust = 0.5)) +
      # animation stuff
      transition_time(frameId) +
      ease_aes('linear') + 
      NULL
    
    play_length <- length(unique(ex_track$frameId))
    play_anim <- animate(
      play_frames,
      fps = 10, 
      nframe = play_length,
      width = 750,
      height = 450,
      end_pause = 0
    )
  } else if(method == 'frame'){
    #ex_track <- ex_track %>% filter(event == event)
    play_anim <- plot_field + 
      # field control
      geom_raster(
        data = ex_play_control, 
        mapping = aes(x = x, y = y, fill = control), alpha = 0.5, interpolate = T
      ) +
      scale_fill_gradient2(
        low = ex_game$away_1, high = ex_game$home_2, mid = "white", midpoint = 0.5, 
        name = "Team Field Control", limits = c(0,1), breaks = c(0, 1), labels = c(ex_game$visitorTeamAbbr, ex_game$homeTeamAbbr)
      ) +  
      # line of scrimmage
      annotate(
        "segment",
        x = line_of_scrimmage, xend = line_of_scrimmage, y = 0, yend = 160/3,
        colour = "#0d41e1", size = 1.5
      ) +
      # away team locs and jersey numbers
      geom_point(
        data = ex_track %>% filter(team == ex_game$visitorTeamAbbr),
        mapping = aes(x = x, y = y),
        fill = "#f8f9fa", colour = ex_game$away_2,
        shape = 21, alpha = 1, size = 8, stroke = 1.5
      ) +
      geom_text(
        data = ex_track %>% filter(team == ex_game$visitorTeamAbbr),
        mapping = aes(x = x, y = y, label = jerseyNumber),
        colour = ex_game$away_1, size = 4
      ) +
      # home team locs and jersey numbers
      geom_point(
        data = ex_track %>% filter(team == ex_game$homeTeamAbbr),
        mapping = aes(x = x, y = y),
        fill = ex_game$home_2, colour = ex_game$home_1,
        shape = 21, alpha = 1, size = 8, stroke = 1.5
      ) +
      # rules-based position and player name
      geom_text(
        data = ex_track %>% filter(team == ex_game$homeTeamAbbr),
        mapping = aes(x = x, y = y, label = jerseyNumber),
        colour = ex_game$home_1, size = 4,
      ) +
      # ball
      geom_point(
        data = ex_track %>% filter(team == "football"),
        mapping = aes(x = x, y = y),
        fill = "#935e38", colour = "#d9d9d9",
        shape = 21, alpha = 1, size = 4, stroke = 1
      ) +
      # title 
      labs(title = play_title,
           subtitle = ex_play$playDescription) +
      theme(plot.title = element_text(face = "bold"),
            plot.subtitle = element_text(hjust = 0.5))
  } else if(method == 'field_control_basic' || method=='field_control_spacevalue'){
    ex_track <- ex_track %>% filter(frameId %in% unique(ex_play_control$frameId))
    
    play_frames <- plot_field + 
      # field control
      geom_raster(
        data = ex_play_control, 
        mapping = aes(x = x, y = y, fill = control), alpha = 0.8, interpolate = T
      ) +
      scale_fill_gradient2(
        low = ex_game$away_1, high = ex_game$home_2, mid = "white", midpoint = 0.5, 
        name = "Team Field Control", limits = c(0,1), breaks = c(0, 1), labels = c(ex_game$visitorTeamAbbr, ex_game$homeTeamAbbr)
      ) +
      # line of scrimmage
      annotate(
        "segment",
        x = line_of_scrimmage, xend = line_of_scrimmage, y = 0, yend = 160/3,
        colour = "#0d41e1", size = 1.5
      ) +
     # away team velocities
      geom_segment(
        data = ex_track %>% filter(team == ex_game$visitorTeamAbbr),
        mapping = aes(x = x, y = y, xend = x + v_x, yend = y + v_y),
        colour = ex_game$away_1, size = 1, arrow = arrow(length = unit(0.01, "npc"))
      ) + 
      # home team velocities
      geom_segment(
        data = ex_track %>% filter(team == ex_game$homeTeamAbbr),
        mapping = aes(x = x, y = y, xend = x + v_x, yend = y + v_y),
        colour = ex_game$home_2, size = 1, arrow = arrow(length = unit(0.01, "npc"))
      ) +
      # away team locs and jersey numbers
      geom_point(
        data = ex_track %>% filter(team == ex_game$visitorTeamAbbr),
        mapping = aes(x = x, y = y),
        fill = "#f8f9fa", colour = ex_game$away_2,
        shape = 21, alpha = 1, size = 8, stroke = 1.5
      ) +
      geom_text(
        data = ex_track %>% filter(team == ex_game$visitorTeamAbbr),
        mapping = aes(x = x, y = y, label = jerseyNumber),
        colour = ex_game$away_1, size = 4.5
      ) +
      # home team locs and jersey numbers
      geom_point(
        data = ex_track %>% filter(team == ex_game$homeTeamAbbr),
        mapping = aes(x = x, y = y),
        fill = ex_game$home_1, colour = ex_game$home_2,
        shape = 21, alpha = 1, size = 8, stroke = 1.5
      ) +
      geom_text(
        data = ex_track %>% filter(team == ex_game$homeTeamAbbr),
        mapping = aes(x = x, y = y, label = jerseyNumber),
        colour = ex_game$home_2, size = 4.5, 
      ) +
      # ball
      geom_point(
        data = ex_track %>% filter(team == "football"),
        mapping = aes(x = x, y = y),
        fill = "#935e38", colour = "#d9d9d9",
        shape = 21, alpha = 1, size = 4, stroke = 1
      ) +
      # title 
      labs(title = play_title,
           subtitle = ex_play$playDescription) +
      theme(plot.title = element_text(face = "bold"),
            plot.subtitle = element_text(hjust = 0.5)) +
      # animation stuff
      transition_time(frameId) +
      ease_aes('linear') + 
      NULL
    
    play_length <- length(unique(ex_track$frameId))
    play_anim <- animate(
      play_frames,
      fps = 10, 
      nframe = play_length,
      width = 750,
      height = 450,
      end_pause = 0
    )
  }
  
  return(play_anim)
}


plot_field <- function(field_color = "#009A17", # #ffffff
                       line_color = "#ffffff", # #212529
                       number_color = "#ffffff") { # #adb5bd
  
  field_height <- 160/3
  field_width <- 120
  
  field <- ggplot() +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 13, hjust = 0.5),
      plot.subtitle = element_text(hjust = 1),
      legend.position = "bottom",
      legend.title.align = 1,
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.title = element_blank(),
      axis.ticks = element_blank(),
      axis.text = element_blank(),
      axis.line = element_blank(),
      panel.background = element_rect(fill = alpha(field_color, 0.5), color = "white"),
      panel.border = element_blank(),
      aspect.ratio = field_height/field_width
    ) +
    # major lines
    annotate(
      "segment",
      x = c(0, 0, 0,field_width, seq(10, 110, by=5)),
      xend = c(field_width,field_width, 0, field_width, seq(10, 110, by=5)),
      y = c(0, field_height, 0, 0, rep(0, 21)),
      yend = c(0, field_height, field_height, field_height, rep(field_height, 21)),
      colour = line_color
    ) +
    # hashmarks
    annotate(
      "segment",
      x = rep(seq(10, 110, by = 1), 4),
      xend = rep(seq(10, 110, by = 1), 4),
      y = c(rep(0, 101), rep(field_height - 1, 101), rep(160/6 + 18.5/6, 101), rep(160/6 - 18.5/6, 101)),
      yend = c(rep(1, 101), rep(field_height, 101), rep(160/6 + 18.5/6 + 1, 101), rep(160/6 - 18.5/6 - 1, 101)),
      colour = line_color
    ) +
    # yard numbers
    annotate(
      "text",
      x = seq(20, 100, by = 10),
      y = rep(12, 9),
      label = c(seq(10, 50, by = 10), rev(seq(10, 40, by = 10))),
      size = 10,
      colour = number_color,
    ) +
    # yard numbers upside down
    annotate(
      "text",
      x = seq(20, 100, by = 10),
      y = rep(field_height - 12, 9),
      label = c(seq(10, 50, by = 10), rev(seq(10, 40, by = 10))),
      angle = 180,
      size = 10,
      colour = number_color, 
    )
  
  return(field)
}
